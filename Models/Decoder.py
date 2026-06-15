import torch
import torch.nn as nn

import Models.Common as Common

class ESPERNetDecoder(nn.Module):
    def __init__(self,
                 output_dim:int=98, # pitch (1) + voiced (33) + unvoiced (257/4=64)
                 pitch_embed_dim:int=8,
                 pos_embed_dim:int=8,
                 max_ctx_size:int=4096,
                 model_dim:int=512,
                 voice_dim:int=64,
                 phoneme_dim:int=3
                 ):
        super().__init__()
        self.output_dim = output_dim
        self.pitch_embed_dim = pitch_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.max_ctx_size = max_ctx_size
        self.model_dim = model_dim
        self.voice_dim = voice_dim
        self.phoneme_dim = phoneme_dim
        self.pre_projector = nn.Linear(phoneme_dim + pitch_embed_dim + pos_embed_dim + voice_dim, model_dim)
        self.main_decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, 8, batch_first=True), num_layers=4)
        self.post_projector = nn.Linear(model_dim, output_dim - 1)

    def forward(self, voice: torch.Tensor, pitch, phoneme):
        assert voice.ndim == 2, f"Voice features must be 2D (batch, channels). Got {voice.ndim}D instead."
        assert pitch.ndim == 2, f"Pitch features must be 2D (batch, time). Got {pitch.ndim}D instead."
        assert phoneme.ndim == 3, f"Phoneme features must be 3D (batch, time, channels). Got {phoneme.ndim}D instead."
        assert voice.shape[1] == self.voice_dim, f"Expected voice features to have {self.voice_dim} channels, got {voice.shape[1]} instead."
        assert phoneme.shape[2] == self.phoneme_dim, f"Expected phoneme features to have {self.phoneme_dim} channels, got {phoneme.shape[2]} instead."
        assert voice.shape[0] == pitch.shape[0] == phoneme.shape[0],\
            f"Batch size mismatch between voice, pitch and phoneme features: Got {voice.shape[0]}, {pitch.shape[0]} and {phoneme.shape[0]}."
        assert voice.device == pitch.device == phoneme.device,\
            f"Expected voice, pitch and phoneme features to be on the same device: Got {voice.device}, {pitch.device} and {phoneme.device}"
        assert pitch.shape[1] == phoneme.shape[1], f"Sequence length mismatch pitch and phoneme features: Got {pitch.shape[1]} and {phoneme.shape[1]}."
        assert phoneme.shape[1] <= self.max_ctx_size, f"Input sequence exceeds max context size. Expected <={self.max_ctx_size}, got {phoneme.shape[1]} tokens."

        batch_size = phoneme.shape[0]
        seq_len = phoneme.shape[1]
        pitch_embedding = Common.pitch_embedding(pitch, size=self.pitch_embed_dim)
        pos_embedding = Common.position_embedding(batch_size, seq_len, self.max_ctx_size, self.pos_embed_dim)
        pos_embedding = pos_embedding.to(phoneme.device)
        voice = voice.unsqueeze(1).repeat(1, seq_len, 1)
        features = torch.cat([phoneme, pitch_embedding, pos_embedding, voice], dim=2)
        features = self.pre_projector(features)
        
        # Create causal mask for autoregressive decoding
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=features.device), diagonal=1)
        features = self.main_decoder(features, mask=causal_mask)
        output = self.post_projector(features)
        output = torch.cat([pitch[..., None], output], dim=2)
        return output

if __name__ == "__main__":
    model = ESPERNetDecoder()
    print(model)
    # print the number of model parameters
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(f"Number of parameters: {params:,}")
    # test inference
    model.eval()
    vo = torch.randn(1, 64)
    pi = torch.randn(1, 1024)
    ph = torch.randn(1, 1024, 3)
    x = model(vo, pi, ph)
    print(x.shape)
