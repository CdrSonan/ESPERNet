from pathlib import Path

import torch

from Models.Classifier import ESPERNetClassifier
from Models.Decoder import ESPERNetDecoder
from Models.Encoder import ESPERNetEncoder

encoder = ESPERNetEncoder().eval()
decoder = ESPERNetDecoder().eval()
classifier = ESPERNetClassifier().eval()

encoder.load_state_dict(torch.load("../models/ESPERNetEncoder.pth", map_location="cpu"))
decoder.load_state_dict(torch.load("../models/ESPERNetDecoder.pth", map_location="cpu"))
classifier.load_state_dict(torch.load("../models/ESPERNetClassifier.pth", map_location="cpu"))

encoder_path = Path("../models/ESPERNetEncoder.onnx")
decoder_path = Path("../models/ESPERNetDecoder.onnx")
classifier_path = Path("../models/ESPERNetClassifier.onnx")
encoder_input = (torch.randn(1, 1024, 291), torch.tensor([1.0,]))
decoder_input = (torch.randn(1, 64), torch.randn(1, 1024), torch.randn(1, 1024, 5))
classifier_input = (torch.randn(1, 1024, 291),)

batch_size = torch.export.Dim("batch_size")
seq_len = torch.export.Dim("seq_len")

torch.onnx.export(
    encoder,
    encoder_input,
    encoder_path,
    input_names=["ESPERAudio", "sampling_factor"],
    output_names=["voice", "pitch", "phoneme"],
    dynamic_shapes={"x": {0: batch_size, 1:seq_len}, "sampling_factor": {0: batch_size}},
    export_params=True,
    do_constant_folding=False,
    external_data=False,
    opset_version=21
)
torch.onnx.export(
    decoder,
    decoder_input,
    decoder_path,
    input_names=["voice", "pitch", "phoneme"],
    output_names=["ESPERAudio"],
    dynamic_shapes={"voice": {0: batch_size}, "pitch": {0: batch_size, 1: seq_len}, "phoneme": {0: batch_size, 1:seq_len}},
    export_params=True,
    do_constant_folding=False,
    external_data=False,
    opset_version=21
)
torch.onnx.export(
    classifier,
    classifier_input,
    classifier_path,
    input_names=["ESPERAudio"],
    output_names=["score"],
    dynamic_shapes={"x": {0: batch_size, 1:seq_len}},
    export_params=True,
    do_constant_folding=False,
    external_data=False,
    opset_version=21
)