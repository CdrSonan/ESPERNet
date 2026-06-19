import math

import torch
from tqdm import tqdm

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

start = 0

#encoder.load_state_dict(torch.load(f"./ESPERNetEncoder-{start}.pth", map_location=device))
#decoder.load_state_dict(torch.load(f"./ESPERNetDecoder-{start}.pth", map_location=device))
# classifier.load_state_dict(torch.load(f"./ESPERNetClassifier-{start}.pth", map_location=device))

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
counter = start
for e in range(2):
    for i, sample in enumerate(tqdm(dataset)):
        batch = sample.to(device)
        loss_stats = scaffold.train_step(batch, float(i) / length if e == 0 else 1, math.sqrt(float(i) / length) if e == 0 else 1)
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
        if counter == start + (e + 1) * length:
            break
    encoder_optimizer.lr = 2e-5
    decoder_optimizer.lr = 2e-5
    classifier_optimizer.lr = 1e-5
torch.save(encoder.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
torch.save(decoder.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
torch.save(classifier.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
torch.save(encoder_optimizer.state_dict(), f"./ESPERNetEncoderOptimizer-{counter}.pth")
torch.save(decoder_optimizer.state_dict(), f"./ESPERNetDecoderOptimizer-{counter}.pth")
torch.save(classifier_optimizer.state_dict(), f"./ESPERNetClassifierOptimizer-{counter}.pth")
