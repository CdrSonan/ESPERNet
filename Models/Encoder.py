import torch
import torch.nn as nn

import Models.Common as Common

class ESPERNetEncoder(nn.Module):
    def __init__(self,
                 input_dim: int=291, # pitch (1) + voiced (33) + unvoiced (257)
                 pitch_embed_dim: int=8,
                 pos_embed_dim: int=8,
                 max_ctx_size: int=1024,
                 model_dim: int=512,
                 voice_dim: int=64,
                 phoneme_dim: int=5
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.pitch_embed_dim = pitch_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.max_ctx_size = max_ctx_size
        self.model_dim = model_dim
        self.voice_dim = voice_dim
        self.phoneme_dim = phoneme_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pre_projector = nn.Linear(input_dim + pitch_embed_dim + pos_embed_dim - 1, model_dim)
        self.main_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, 8, batch_first=True), num_layers=6)
        self.post_projector_voice = nn.Linear(model_dim, voice_dim * 2) # Multiplier 2 due to VAE prediction (mean + variance)
        self.post_projector_phoneme = nn.Linear(model_dim, phoneme_dim * 2)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3, f"Input must be 3D (batch, time, channels). Got {x.ndim}D instead."
        assert x.shape[2] == self.input_dim, f"Expected input to have {self.input_dim} channels, got {x.shape[2]} instead."
        assert x.shape[1] <= self.max_ctx_size, f"Input sequence exceeds max context size. Expected <={self.max_ctx_size}, got {x.shape[1]} tokens."

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        pitch = x[..., 0]
        features = x[..., 1:]
        pitch_embedding = Common.pitch_embedding(pitch, size=self.pitch_embed_dim)
        pos_embedding = Common.position_embedding(batch_size, seq_len, self.max_ctx_size, self.pos_embed_dim)
        pos_embedding = pos_embedding.to(x.device)
        features = torch.cat([features, pitch_embedding, pos_embedding], dim=2)
        features = self.pre_projector(features)
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([features, cls_token_expanded], dim=1)
        features = self.main_encoder(features, mask=self.attn_mask(seq_len + 1, device=features.device))
        voice_features = features[:, -1, :]
        phoneme_features = features[:, :-1, :]
        voice_features = self.post_projector_voice(voice_features)
        phoneme_features = self.post_projector_phoneme(phoneme_features)
        return voice_features, pitch, phoneme_features

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
    model.eval().cuda()
    data = torch.randn(1, 1024, 291).cuda()
    u, v, w = model(data)
    print(u.shape, v.shape, w.shape)