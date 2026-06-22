import math

import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np

from Models.Classifier import ESPERNetClassifier
from Models.Decoder import ESPERNetDecoder
from Models.Encoder import ESPERNetEncoder
from Training.Loss_Module import BatchInvariantVAELoss
from Training.StreamingDataset import EsperServerDataset
from Training.Training_Scaffold import ESPERNetTrainingScaffold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = ESPERNetEncoder().to(device)
decoder = ESPERNetDecoder().to(device)
classifier = ESPERNetClassifier().to(device)

encoder_optimizer = torch.optim.NAdam(encoder.parameters(), lr=1e-4)
decoder_optimizer = torch.optim.NAdam(decoder.parameters(), lr=1e-4)
classifier_optimizer = torch.optim.NAdam(classifier.parameters(), lr=1e-5)
loss_module = BatchInvariantVAELoss(latent_dim=3)
scaffold = ESPERNetTrainingScaffold(encoder,
                                    decoder,
                                    classifier,
                                    encoder_optimizer,
                                    decoder_optimizer,
                                    classifier_optimizer,
                                    loss_module)

dataset = EsperServerDataset(address="tcp://192.168.1.116:5555")
length = len(dataset)
length_warmup = length // 4
length_calibrate = 1000
length_convert = length
length_winddown = 3 * length // 4

phoneme_samples = None

counter = 0
for i, sample in enumerate(tqdm(dataset, total=length_warmup)):
    batch = sample.to(device)
    loss_stats = scaffold.train_step(batch, 0, 0)
    counter += 1
    if counter % 50 == 0:
        print(loss_stats)
    if counter % 5000 == 0:
        torch.save(encoder.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
        torch.save(decoder.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
        torch.save(classifier.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
        torch.save(encoder_optimizer.state_dict(), f"./ESPERNetEncoderOptimizer-{counter}.pth")
        torch.save(decoder_optimizer.state_dict(), f"./ESPERNetDecoderOptimizer-{counter}.pth")
        torch.save(classifier_optimizer.state_dict(), f"./ESPERNetClassifierOptimizer-{counter}.pth")
    if i == length_warmup:
        break

with torch.no_grad():
    for i, sample in enumerate(tqdm(dataset, total=length_calibrate)):
        batch = sample.to(device)
        voice, pitch, phoneme = scaffold.encoder(
            batch,
            torch.ones(batch.shape[0], device=batch.device),
            torch.zeros(batch.shape[0], device=batch.device),
            return_stats=False,
        )
        phoneme_flat = phoneme.detach().to(torch.device("cpu")).reshape(-1, phoneme.shape[-1])
        if phoneme_samples is None:
            phoneme_samples = phoneme_flat
        else:
            phoneme_samples = torch.cat([phoneme_samples, phoneme_flat], dim=0)
        if i == length_calibrate:
            break
kmeans = KMeans(n_clusters=scaffold.encoder.codebook_size, random_state=0)
kmeans.fit(phoneme_samples.numpy())
print(kmeans.cluster_centers_)
scaffold.encoder.codebook.weight.data = torch.from_numpy(kmeans.cluster_centers_).to(device)
scaffold.loss.lambda_kl = 0.0
scaffold.loss.lambda_var = 0.0

for i, sample in enumerate(tqdm(dataset, total=length_convert)):
    schedule = math.sin((math.pi / 2) * (i / length_convert))
    batch = sample.to(device)
    loss_stats = scaffold.train_step(batch, schedule, 1)
    counter += 1
    if counter % 50 == 0:
        print(loss_stats)
    if counter % 5000 == 0:
        torch.save(encoder.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
        torch.save(decoder.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
        torch.save(classifier.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
        torch.save(encoder_optimizer.state_dict(), f"./ESPERNetEncoderOptimizer-{counter}.pth")
        torch.save(decoder_optimizer.state_dict(), f"./ESPERNetDecoderOptimizer-{counter}.pth")
        torch.save(classifier_optimizer.state_dict(), f"./ESPERNetClassifierOptimizer-{counter}.pth")
    if i == length_convert:
        break

for i, sample in enumerate(tqdm(dataset, total=length_winddown)):
    schedule = 0.5 * (1.0 + math.cos(math.pi * (i / length_winddown)))
    encoder_optimizer.lr = 1e-4 * schedule
    decoder_optimizer.lr = 1e-4 * schedule
    classifier_optimizer.lr = 1e-5 * schedule
    batch = sample.to(device)
    loss_stats = scaffold.train_step(batch, 1, 1)
    counter += 1
    if counter % 50 == 0:
        print(loss_stats)
    if counter % 5000 == 0:
        torch.save(encoder.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
        torch.save(decoder.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
        torch.save(classifier.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
        torch.save(encoder_optimizer.state_dict(), f"./ESPERNetEncoderOptimizer-{counter}.pth")
        torch.save(decoder_optimizer.state_dict(), f"./ESPERNetDecoderOptimizer-{counter}.pth")
        torch.save(classifier_optimizer.state_dict(), f"./ESPERNetClassifierOptimizer-{counter}.pth")
    if i == length_winddown:
        break

torch.save(encoder.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
torch.save(decoder.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
torch.save(classifier.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
torch.save(encoder_optimizer.state_dict(), f"./ESPERNetEncoderOptimizer-{counter}.pth")
torch.save(decoder_optimizer.state_dict(), f"./ESPERNetDecoderOptimizer-{counter}.pth")
torch.save(classifier_optimizer.state_dict(), f"./ESPERNetClassifierOptimizer-{counter}.pth")