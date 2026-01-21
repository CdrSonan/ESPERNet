from typing import List

import torch

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

if __name__ == "__main__":
    # test training step
    encoder = ESPERNetEncoder()
    decoder = ESPERNetDecoder()
    classifier = ESPERNetClassifier()
    encoder_optimizer = torch.optim.Adam(encoder.parameters())
    decoder_optimizer = torch.optim.Adam(decoder.parameters())
    classifier_optimizer = torch.optim.Adam(classifier.parameters())
    scaffold = ESPERNetTrainingScaffold(encoder, decoder, classifier, encoder_optimizer, decoder_optimizer, classifier_optimizer, torch.nn.MSELoss())
    batch = torch.randn(1, 1024, 291)
    vae_loss, gan_loss_decoder, gan_loss_classifier = scaffold.train_step(batch)
    print(vae_loss, gan_loss_decoder, gan_loss_classifier)
    print("Training done!")
