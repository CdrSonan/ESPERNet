
import math
import torch
def pitch_embedding(pitch: torch.Tensor, lower_bound:float=40.0, upper_bound:float=2000.0, size:int=8):
    lower_bound_log = math.log2(lower_bound)
    upper_bound_log = math.log2(upper_bound)
    pitch_log = torch.log2(pitch) # deliberately produces NaN for 0 values
    coord = (pitch_log - lower_bound_log) / (upper_bound_log - lower_bound_log)
    coord = coord.unsqueeze(-1)
    embedding = torch.cat([coord * (i + 1) for i in range(size)], dim=-1)
    embedding = torch.cos(embedding * math.pi)
    embedding = embedding.nan_to_num(nan=0.0) # special case for 0 values
    return embedding

def position_embedding(batch_size: int, length: int, max_ctx_size: int = 1024, size: int = 8):
    coord = (torch.arange(length, dtype=torch.float32)[None, ...] / max_ctx_size).expand(batch_size, -1)
    embedding = torch.cat([coord[..., None] * (i + 1) for i in range(size)], dim=2)
    embedding = torch.sin(embedding * math.pi)
    return embedding
