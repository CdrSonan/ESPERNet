import torch

from Models.Classifier import ESPERNetClassifier
from Models.Decoder import ESPERNetDecoder
from Models.Encoder import ESPERNetEncoder
from Training.Loss_Module import BatchInvariantVAELoss


class ESPERNetTrainingScaffold:
    def __init__(self,
                 encoder: ESPERNetEncoder,
                 decoder: ESPERNetDecoder,
                 classifier: ESPERNetClassifier,
                 encoder_optimizer: torch.optim.Optimizer,
                 decoder_optimizer: torch.optim.Optimizer,
                 classifier_optimizer: torch.optim.Optimizer,
                 loss: BatchInvariantVAELoss,
                 gan_weight: float = 0.0):

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.classifier_optimizer = classifier_optimizer
        self.loss = loss
        self.gan_weight = gan_weight

    def train_step(self, batch: torch.Tensor):
        voice, pitch, phoneme, voice_mean, voice_logvar, phoneme_mean, phoneme_logvar = self.encoder(
            batch,
            torch.ones(batch.shape[0], device=batch.device),
            return_stats=True,
        )

        decoded = self.decoder(voice, pitch, phoneme)

        vae_loss_total, stats = self.loss(phoneme_mean, phoneme_logvar, decoded, batch)
        score_generator = self.classifier(decoded)
        gan_loss_decoder = torch.abs(score_generator).mean()

        (vae_loss_total + self.gan_weight * gan_loss_decoder).backward()
        vae_loss_total.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()

        score_real = self.classifier(batch)
        score_fake = self.classifier(decoded.detach())
        gan_loss_classifier = torch.square(score_real).mean() + torch.square(score_fake - 1).mean()
        gan_loss_classifier.backward()
        self.classifier_optimizer.step()

        self.classifier_optimizer.zero_grad()

        stats["gan_d"] = gan_loss_decoder
        stats["gan_c"] = gan_loss_classifier

        return stats

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
    loss_stats = scaffold.train_step(batch)
    print(loss_stats)
    print("Training done!")
