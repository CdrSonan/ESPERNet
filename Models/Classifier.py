from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import Models.Common as Common

class ESPERNetClassifier(nn.Module):
    def __init__(self,
                 input_dim:int=98, # pitch (1) + voiced (33) + unvoiced (257/4=64)
                 n_voiced:int=33,
                 filter_counts:List[int]=[8, 16, 32, 64]
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.n_voiced = n_voiced
        self.dims = [2] + filter_counts
        self.filters = nn.Sequential(
            *[
                self.make_filter(in_channels, out_channels) for
                in_channels, out_channels in
                zip(self.dims[:-1], self.dims[1:])
            ]
        )
        self.projector = nn.Linear(self.dims[-1], 1)

    def make_filter(self, in_channels:int, out_channels:int):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)),
            nn.GELU(),
        )

    def forward(self, x:torch.Tensor):
        assert x.ndim == 3, f"Input must be 3D (batch, time, channels). Got {x.ndim}D instead."
        assert x.shape[2] == self.input_dim, f"Expected input to have {self.input_dim} channels, got {x.shape[2]} instead."


        features = x[..., 1:]
        voiced_hint = torch.zeros_like(features)
        voiced_hint[..., :self.n_voiced] = 1
        features = torch.stack([features, voiced_hint], dim=1)
        # shape: (batch, 2, time, channels)
        # CNN treats time and channels as x and y
        result = self.filters(features)
        means = torch.mean(result, dim=(1, 2, 3))
        return means

if __name__ == "__main__":
    model = ESPERNetClassifier()
    print(model)
    # print the number of model parameters
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(f"Number of parameters: {params:,}")
    # test inference
    model.eval()
    data = torch.randn(4, 1024, 98)
    out = model(data)
    print(out.shape)