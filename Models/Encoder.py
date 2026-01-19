import math
import torch
import torch.nn as nn

class ESPERNetEncoder(nn.Module):
    def __init__(self,
                 input_dim:int=291, # pitch (1) + voiced (33) + unvoiced (257)
                 pitch_embed_dim:int=8,
                 pos_embed_dim:int=8,
                 max_ctx_size:int=1024,
                 model_dim:int=512,
                 voice_dim:int=64,
                 phoneme_dim:int=5
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
        self.voice_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, 8, batch_first=True), num_layers=6)
        self.phoneme_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, 8, batch_first=True), num_layers=6)
        self.post_projector_voice = nn.Linear(model_dim, voice_dim)
        self.post_projector_phoneme = nn.Linear(model_dim, phoneme_dim)

    def forward(self, x):
        assert x.ndim == 3, f"Input must be 3D (batch, time, channels. Got{x.ndim}D instead."
        assert x.shape[1] <= self.max_ctx_size, f"Input sequence exceeds max context size. Expected <={self.max_ctx_size}, got {x.shape[1]} tokens."
        assert x.shape[2] == self.input_dim, f"Input channel count must match (voiced_dim + unvoiced_dim + 1). Expected{self.input_dim}, got {x.shape[2]} instead."

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        pitch = x[..., 0]
        features = x[..., 1:]
        pitch_embedding = self.pitch_embedding(pitch, size=self.pitch_embed_dim)
        pos_embedding = self.position_embedding(batch_size, seq_len, self.max_ctx_size, self.pos_embed_dim)
        features = torch.cat([features, pitch_embedding, pos_embedding], dim=-1)
        features = self.pre_projector(features)
        phoneme_features = self.phoneme_encoder(features)
        features = torch.cat([features, self.cls_token], dim=1)
        voice_features = self.voice_encoder(features)[:, -1, :]
        phoneme_features = self.post_projector_phoneme(phoneme_features)
        voice_features = self.post_projector_voice(voice_features)
        return voice_features, phoneme_features

    @staticmethod
    def pitch_embedding(pitch: torch.Tensor, lower_bound:float=40.0, upper_bound:float=2000.0, size:int=8):
        lower_bound_log = math.log2(lower_bound)
        upper_bound_log = math.log2(upper_bound)
        pitch_log = torch.log2(pitch) # deliberately produces NaN for 0 values
        coord = (pitch_log - lower_bound_log) / (upper_bound_log - lower_bound_log)
        coord = coord.unsqueeze(-1)
        embedding = torch.cat([coord * i for i in range(size)], dim=-1)
        embedding = torch.cos(embedding * math.pi)
        embedding = embedding.nan_to_num(nan=0.0) # special case for 0 values
        return embedding

    @staticmethod
    def position_embedding(batch_size: int, length: int, max_ctx_size: int = 1024, size: int = 8):
        coord = torch.cat([torch.arange(length, dtype=torch.float32)[None, ...] / max_ctx_size for i in range(batch_size)], dim=1)
        embedding = torch.cat([coord[..., None] * i for i in range(size)], dim=2)
        embedding = torch.sin(embedding * math.pi)
        return embedding

if __name__ == "__main__":
    model = ESPERNetEncoder()
    print(model)
    # print the number of model parameters
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(f"Number of parameters: {params:,}")
    # test inference
    x = torch.randn(1, 1024, 291)
    y, z = model(x)
    print(y.shape, z.shape)