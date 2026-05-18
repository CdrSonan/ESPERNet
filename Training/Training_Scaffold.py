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
                 vae_loss_fn: torch.nn.Module,
                 kl_weight: float = 1e-4,
                 phoneme_con_weight: float = 1.0):

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.classifier_optimizer = classifier_optimizer
        self.vae_loss_fn = vae_loss_fn
        self.kl_weight = kl_weight
        self.phoneme_con_weight = phoneme_con_weight

    @staticmethod
    def _kl_divergence_standard_normal(mean: torch.Tensor, logvar: torch.Tensor):
        return -0.5 * (1.0 + logvar - mean.square() - torch.exp(logvar)).mean()

    @staticmethod
    def _gaussian_pair_nll(mean_ref: torch.Tensor, logvar_ref: torch.Tensor, mean_aug: torch.Tensor, logvar_aug: torch.Tensor):
        var_ref = torch.exp(logvar_ref)
        var_aug = torch.exp(logvar_aug)
        combined_var = var_ref + var_aug
        mean_delta_sq = (mean_aug - mean_ref).square()
        return 0.5 * (mean_delta_sq / (combined_var + 1e-8) + torch.log(combined_var + 1e-8)).mean()

    def train_step(self, batch: torch.Tensor):
        voice, pitch, phoneme, voice_mean, voice_logvar, phoneme_mean, phoneme_logvar = self.encoder(
            batch,
            torch.ones(batch.shape[0], device=batch.device),
            return_stats=True,
        )

        batch_size = phoneme_mean.shape[0]
        if batch_size > 1:
            phoneme_ref_mean, phoneme_aug_mean = phoneme_mean.split_with_sizes([1, batch_size - 1], dim=0)
            phoneme_ref_logvar, phoneme_aug_logvar = phoneme_logvar.split_with_sizes([1, batch_size - 1], dim=0)
            phoneme_con_loss = self._gaussian_pair_nll(
                phoneme_ref_mean.expand_as(phoneme_aug_mean),
                phoneme_ref_logvar.expand_as(phoneme_aug_logvar),
                phoneme_aug_mean,
                phoneme_aug_logvar,
            )
        else:
            phoneme_con_loss = torch.zeros((), device=batch.device)

        decoded = self.decoder(voice, pitch, phoneme)

        vae_loss = self.vae_loss_fn(batch, decoded)
        kl_loss = self._kl_divergence_standard_normal(voice_mean, voice_logvar) + self._kl_divergence_standard_normal(phoneme_mean, phoneme_logvar)
        #score_generator = self.classifier(decoded)
        #gan_loss_decoder = torch.abs(score_generator).mean()

        (vae_loss + self.phoneme_con_weight * phoneme_con_loss + self.kl_weight * kl_loss).backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        #self.classifier_optimizer.zero_grad()

        """score_real = self.classifier(batch)
        score_fake = self.classifier(decoded.detach())
        gan_loss_classifier = torch.square(score_real).mean() + torch.square(score_fake - 1).mean()
        gan_loss_classifier.backward()
        self.classifier_optimizer.step()

        self.classifier_optimizer.zero_grad()"""

        gan_loss_classifier = torch.tensor([0,])

        return vae_loss.item(), phoneme_con_loss.item(), kl_loss.item(), gan_loss_classifier.item()

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
    vae_loss, phoneme_con_loss, kl_loss, gan_loss_classifier = scaffold.train_step(batch)
    print(vae_loss, phoneme_con_loss, kl_loss, gan_loss_classifier)
    print("Training done!")
