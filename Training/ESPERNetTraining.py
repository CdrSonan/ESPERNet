import torch
from tqdm import tqdm

from Models.Classifier import ESPERNetClassifier
from Models.Decoder import ESPERNetDecoder
from Models.Encoder import ESPERNetEncoder
from Training.Loss_Module import VAELoss
from Training.StreamingDataset import EsperServerDataset
from Training.Training_Scaffold import ESPERNetTrainingScaffold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = ESPERNetEncoder().to(device)
decoder = ESPERNetDecoder().to(device)
classifier = ESPERNetClassifier().to(device)
encoder_optimizer = torch.optim.NAdam(encoder.parameters(), lr=1e-4)
decoder_optimizer = torch.optim.NAdam(decoder.parameters(), lr=1e-4)
classifier_optimizer = torch.optim.NAdam(classifier.parameters(), lr=1e-5)
loss_module = VAELoss(latent_dim=3)
scaffold = ESPERNetTrainingScaffold(encoder,
                                    decoder,
                                    classifier,
                                    encoder_optimizer,
                                    decoder_optimizer,
                                    classifier_optimizer,
                                    loss_module)

dataset = EsperServerDataset(address="tcp://192.168.1.116:5555")
length = len(dataset)
counter = 0
for sample in tqdm(dataset):
    batch = sample.to(device)
    loss_stats = scaffold.train_step(batch)
    counter += 1
    if counter % 10 == 0:
        print(loss_stats)
    if counter % 1000 == 0:
        torch.save(encoder.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
        torch.save(decoder.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
        torch.save(classifier.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
    if counter == length:
        break
