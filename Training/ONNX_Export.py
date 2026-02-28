from pathlib import Path

import onnx
import torch
from torch import nn
#from onnxruntime.training import artifacts

from Models.Classifier import ESPERNetClassifier
from Models.Decoder import ESPERNetDecoder
from Models.Encoder import ESPERNetEncoder

encoder = ESPERNetEncoder().train()
decoder = ESPERNetDecoder().train()
classifier = ESPERNetClassifier().train()
encoder_path = Path("../models/encoder.onnx")
decoder_path = Path("../models/decoder.onnx")
classifier_path = Path("../models/classifier.onnx")
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