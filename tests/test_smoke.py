"""Smoke tests — compile checks and basic model sanity."""

import py_compile
import pathlib
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── Every .py in the repo root must compile ──────────────────────

PY_FILES = sorted(ROOT.glob("*.py"))


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_py_compiles(path):
    py_compile.compile(str(path), doraise=True)


# ── Encoder produces the right shape ─────────────────────────────

def test_encoder_output_shape():
    from training import ProductVisionEncoder

    model = ProductVisionEncoder(embed_size=384)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 384), f"Expected (1, 384), got {out.shape}"


# ── Visual vocabulary builds without error ───────────────────────

def test_vocabulary_builds():
    from inference import build_vocabulary_index, VISUAL_VOCABULARY

    entries, embeddings = build_vocabulary_index()
    total_terms = sum(len(v) for v in VISUAL_VOCABULARY.values())
    assert len(entries) == total_terms
    assert embeddings.shape[0] == total_terms
    assert embeddings.shape[1] == 384
