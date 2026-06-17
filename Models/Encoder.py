import torch
import torch.nn as nn
import torch.nn.functional as F

import Models.Common as Common

class ESPERNetEncoder(nn.Module):
    def __init__(self,
                 input_dim:int=98, # pitch (1) + voiced (33) + unvoiced (257/4=64)
                 pitch_embed_dim: int=8,
                 pos_embed_dim: int=8,
                 max_ctx_size: int=4096,
                 model_dim: int=512,
                 voice_dim: int=64,
                 phoneme_dim: int=3,
                 voice_mean_path_width: int=16,
                 codebook_size: int = 512
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.pitch_embed_dim = pitch_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.max_ctx_size = max_ctx_size
        self.model_dim = model_dim
        self.voice_dim = voice_dim
        self.phoneme_dim = phoneme_dim
        self.voice_mean_path_width = voice_mean_path_width
        self.codebook_size = codebook_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pre_projector = nn.Linear(input_dim + pitch_embed_dim + pos_embed_dim - 1, model_dim)
        self.main_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, 8, batch_first=True), num_layers=2)
        self.post_projector_voice = nn.Linear(model_dim, voice_dim * 2) # Multiplier 2 due to VAE prediction (mean + variance)
        self.post_projector_phoneme = nn.Linear(model_dim, phoneme_dim * 2)

        self.codebook = nn.Embedding(codebook_size, phoneme_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x: torch.Tensor, sampling_factor: torch.Tensor, vq_balance: torch.Tensor, return_stats: bool = False):
        assert x.ndim == 3, f"Input must be 3D (batch, time, channels). Got {x.ndim}D instead."
        assert x.shape[2] == self.input_dim, f"Expected input to have {self.input_dim} channels, got {x.shape[2]} instead."
        assert x.shape[1] <= self.max_ctx_size, f"Input sequence exceeds max context size. Expected <={self.max_ctx_size}, got {x.shape[1]} tokens."
        assert sampling_factor.ndim == 1, f"Sampling factor must be 1D (batch). Got {sampling_factor.ndim}D instead."
        assert sampling_factor.shape[0] == x.shape[0], f"Batch size mismatch between input and sampling factor: Got {x.shape[0]} and {sampling_factor.shape[0]}."

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Construct features tensor:
        # - spectral features (voiced and unvoiced power spectra;
        #   phases are not included since we are processing compressed ESPERAudio)
        # - sinusoidal pitch embedding
        # - sinusoidal position embedding
        pitch = x[..., 0]
        features = x[..., 1:]
        pitch_embedding = Common.pitch_embedding(pitch, size=self.pitch_embed_dim)
        pos_embedding = Common.position_embedding(batch_size, seq_len, self.max_ctx_size, self.pos_embed_dim)
        pos_embedding = pos_embedding.to(x.device)
        features = torch.cat([features, pitch_embedding, pos_embedding], dim=2)

        # project to transformer dimension
        features = self.pre_projector(features)

        # broadcast CLS token to batch size and add to sequence
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([features, cls_token_expanded], dim=1)

        # run the main model
        features = self.main_encoder(features, mask=self.attn_mask(seq_len + 1, device=features.device), is_causal=False)
        #features = self.main_encoder(features)
        voice_features = features[:, -1, :]
        # "voice mean path": a subset of the voice feature is replaced with the mean of a corresponding subset of
        # phoneme features and itself. This gives the model an additional path to transfer information to the CLS token.
        voice_features[:, :self.voice_mean_path_width] = features[:, :, :self.voice_mean_path_width].mean(dim=1)
        phoneme_features = features[:, :-1, :]

        # project to twice the output size
        voice_features = self.post_projector_voice(voice_features)
        phoneme_features = self.post_projector_phoneme(phoneme_features)
        voice_mean, voice_scale_raw = voice_features.chunk(2, dim=-1)
        phoneme_mean, phoneme_scale_raw = phoneme_features.chunk(2, dim=-1)

        # perform vector quantization
        phoneme_flat = phoneme_mean.reshape(-1, self.phoneme_dim)
        distances = (phoneme_flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(dim=2)
        indices = distances.argmin(dim=1)
        phoneme_quantized = self.codebook(indices).reshape(phoneme_mean.shape)
        vq_loss = F.mse_loss(phoneme_quantized.detach(), phoneme_mean)

        voice_std = F.softplus(voice_scale_raw) + 1e-6
        phoneme_std = F.softplus(phoneme_scale_raw) + 1e-6
        voice_logvar = 2.0 * torch.log(voice_std)
        phoneme_logvar = 2.0 * torch.log(phoneme_std)
        voice_sampled = voice_mean + voice_std * torch.randn_like(voice_mean) * sampling_factor[:, None]
        phoneme_sampled = phoneme_mean + phoneme_std * torch.randn_like(phoneme_mean) * sampling_factor[:, None, None]

        phoneme = phoneme_sampled * (torch.ones_like(vq_balance[:, None, None]) - vq_balance[:, None, None]) + phoneme_quantized * vq_balance[:, None, None]

        if return_stats:
            return voice_sampled, pitch, phoneme, voice_mean, voice_logvar, phoneme_mean, phoneme_logvar, vq_loss
        return voice_sampled, pitch, phoneme

    @staticmethod
    def attn_mask(seq_len: int, win_size: int = 7, device: torch.device = torch.device("cpu")):
        # window radius on each side
        half = win_size // 2

        # positions 0 .. seq_len-1, last index is CLS
        idxs = torch.arange(seq_len, device=device)

        # pairwise distance |i - j|
        dist = (idxs[None, :] - idxs[:, None]).abs()

        # base mask: allow only positions within window
        mask = dist > half  # True = masked

        # last token (CLS) can attend to all -> clear its row
        mask[-1, :] = False

        return mask

if __name__ == "__main__":
    model = ESPERNetEncoder()
    print(model)
    # print the number of model parameters
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(f"Number of parameters: {params:,}")
    # test inference
    model.eval()
    data = torch.randn(4, 1024, 98)
    u, v, w = model(data, torch.ones(4, device=data.device))
    print(u.shape, v.shape, w.shape)