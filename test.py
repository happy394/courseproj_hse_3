"""Basic smoke tests for the pipeline components."""

import torch
import numpy as np
from training import ProductVisionEncoder


def test_vision_encoder_output_shape():
    model = ProductVisionEncoder(embed_size=384)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 384), f'expected (2, 384), got {out.shape}'
    print('[PASS] vision encoder output shape')


def test_vision_encoder_deterministic():
    model = ProductVisionEncoder(embed_size=384)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = model(dummy)
        out2 = model(dummy)
    assert torch.allclose(out1, out2), 'outputs differ for same input'
    print('[PASS] vision encoder deterministic')


if __name__ == '__main__':
    test_vision_encoder_output_shape()
    test_vision_encoder_deterministic()
    print('\n[*] all tests passed')
