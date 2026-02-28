from typing import List

import onnx
import torch
from torch import nn
from onnxruntime.training import artifacts

from Models.Classifier import ESPERNetClassifier
from Models.Decoder import ESPERNetDecoder
from Models.Encoder import ESPERNetEncoder


class ESPERNetTrainingScaffold:
    def __init__(self,
                 encoder: ESPERNetEncoder,
                 decoder: ESPERNetDecoder,
                 classifier: ESPERNetClassifier,
                 encoder_optimizer: torch.optim.Optimizer,
                 decoder_optimizer: torch.optim.Optimizer,
                 classifier_optimizer: torch.optim.Optimizer,
                 vae_loss_fn: torch.nn.Module,):

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.classifier_optimizer = classifier_optimizer
        self.vae_loss_fn = vae_loss_fn

    def train_step(self, batch: torch.Tensor):
        voice, pitch, phoneme = self.encoder(batch)
        voice_mean, voice_var = voice.chunk(2, dim=-1)
        phoneme_mean, phoneme_var = phoneme.chunk(2, dim=-1)
        voice_sampled = voice_mean + voice_var * torch.randn_like(voice_mean)
        phoneme_sampled = phoneme_mean + phoneme_var * torch.randn_like(phoneme_mean)
        decoded = self.decoder(voice_sampled, pitch, phoneme_sampled)
        score_generator = self.classifier(decoded)
        vae_loss = self.vae_loss_fn(batch, decoded)
        gan_loss_decoder = torch.abs(score_generator).mean()

        (vae_loss + gan_loss_decoder).backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()

        score_real = self.classifier(batch)
        score_fake = self.classifier(decoded.detach())
        gan_loss_classifier = torch.square(score_real - 1).mean() + torch.square(score_fake).mean()
        gan_loss_classifier.backward()
        self.classifier_optimizer.step()

        self.classifier_optimizer.zero_grad()

        return vae_loss.item(), gan_loss_decoder.item(), gan_loss_classifier.item()

class ESPERNetONNXTrainingGraph(nn.Module):
    def __init__(self,
                 encoder: ESPERNetEncoder,
                 decoder: ESPERNetDecoder,
                 classifier: ESPERNetClassifier,
                 vae_loss_fn: torch.nn.Module):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.vae_loss_fn = vae_loss_fn

    def forward(self, batch: torch.Tensor, vae_mul: torch.Tensor, gan_dec_mul: torch.Tensor, gan_cls_mul: torch.Tensor):
        voice, pitch, phoneme = self.encoder(batch)
        voice_mean, voice_var = voice.chunk(2, dim=-1)
        phoneme_mean, phoneme_var = phoneme.chunk(2, dim=-1)
        voice_sampled = voice_mean + voice_var * torch.randn_like(voice_mean)
        phoneme_sampled = phoneme_mean + phoneme_var * torch.randn_like(phoneme_mean)
        decoded = self.decoder(voice_sampled, pitch, phoneme_sampled)
        score_generator = self.classifier(decoded)
        vae_loss = self.vae_loss_fn(batch, decoded)
        gan_loss_decoder = torch.abs(score_generator)
        score_real = self.classifier(batch)
        score_fake = self.classifier(decoded.detach())
        gan_loss_classifier = torch.square(score_real - 1) + torch.square(score_fake)
        return vae_loss * vae_mul + gan_loss_decoder * gan_dec_mul + gan_loss_classifier * gan_cls_mul

if __name__ == "__main__":
    # ONNX export
    graph = ESPERNetONNXTrainingGraph(
        ESPERNetEncoder(),
        ESPERNetDecoder(),
        ESPERNetClassifier(),
        torch.nn.MSELoss()
    )
    graph.train()
    inputs = (torch.randn(1, 1024, 291), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
    torch.onnx.export(
        graph,
        inputs,
        "model.onnx",
        input_names=["input", "vae_mul", "gan_dec_mul", "gan_cls_mul"],
        output_names=["loss"],
        dynamic_axes={"input": {0: "batch_size", 1:"seq_len"}, "output": {0: "batch_size"}},
        export_params=True,
        do_constant_folding=False
    )
    print("ONNX export successful!")
    model_loaded = onnx.load("model.onnx")
    artifacts.generate_artifacts(
        model_loaded,
        loss = artifacts.LossType.MSELoss,
        optimizer = artifacts.OptimType.AdamW,
        artifact_directory = "./model_artifacts"
    )
    print("Artifacts generated successfully!")
