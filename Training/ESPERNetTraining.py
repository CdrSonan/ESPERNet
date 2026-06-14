import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from Models.Classifier import ESPERNetClassifier
from Models.Decoder import ESPERNetDecoder
from Models.Encoder import ESPERNetEncoder
from Training.Loss_Module import BatchInvariantVAELoss
from Training.StreamingDataset import EsperServerDataset
from Training.Training_Scaffold import ESPERNetTrainingScaffold

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def main():
    # Check if distributed training is enabled
    if 'LOCAL_RANK' in os.environ:
        device = setup_distributed()
        is_distributed = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False

    encoder = ESPERNetEncoder().to(device)
    decoder = ESPERNetDecoder().to(device)
    classifier = ESPERNetClassifier().to(device)
    
    # Wrap models with DDP if distributed training is enabled
    if is_distributed:
        encoder = DDP(encoder, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))
        decoder = DDP(decoder, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))
        classifier = DDP(classifier, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))
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
    counter = 0
    for sample in tqdm(dataset):
        batch = sample.to(device)
        loss_stats = scaffold.train_step(batch)
        counter += 1
        if counter % 10 == 0:
            if is_distributed and dist.get_rank() == 0:
                print(loss_stats)
            elif not is_distributed:
                print(loss_stats)
        if counter % 1000 == 0:
            if is_distributed and dist.get_rank() == 0:
                torch.save(encoder.module.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
                torch.save(decoder.module.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
                torch.save(classifier.module.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
            elif not is_distributed:
                torch.save(encoder.state_dict(), f"./ESPERNetEncoder-{counter}.pth")
                torch.save(decoder.state_dict(), f"./ESPERNetDecoder-{counter}.pth")
                torch.save(classifier.state_dict(), f"./ESPERNetClassifier-{counter}.pth")
        if counter == length:
            break
    
    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    import os
    main()
