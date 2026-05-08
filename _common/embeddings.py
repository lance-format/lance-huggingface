"""GPU-batched embedding encoders used by the conversion scripts.

We deliberately keep things minimal: load the model once, expose batched
``encode_images`` / ``encode_texts`` methods that return float32 numpy arrays
ready to be packed into a Lance ``FixedSizeList`` column.

Embeddings are L2-normalized so that any of cosine/dot/L2 indices give
meaningful results (cosine ~= dot on normalized vectors, and IVF_PQ default
metric ``l2`` works equivalently for nearest-neighbour ranking).
"""

from __future__ import annotations

import io
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# CLIP (image + text)
# ---------------------------------------------------------------------------


class CLIPEncoder:
    """OpenCLIP ViT-B/32 — 512 dim, fast enough for hundreds of thousands of
    images on a single H100.

    The model is loaded lazily so importing this module is free.
    """

    DIM = 512
    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "laion2b_s34b_b79k"

    def __init__(self, device: Optional[str] = None, half: bool = True) -> None:
        import torch
        import open_clip

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.startswith("cuda")

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        model = model.eval().to(self.device)
        if self.half:
            model = model.half()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)

    # -- images -------------------------------------------------------------
    def _to_pil(self, item) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(item)).convert("RGB")
        if isinstance(item, np.ndarray):
            return Image.fromarray(item).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(item)!r}")

    def encode_images(self, images: Sequence, batch_size: int = 256) -> np.ndarray:
        torch = self._torch
        out: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            tensors = self._torch.stack([self.preprocess(self._to_pil(im)) for im in chunk]).to(self.device)
            if self.half:
                tensors = tensors.half()
            with torch.no_grad():
                emb = self.model.encode_image(tensors)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out.append(emb.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.DIM), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)

    # -- text ---------------------------------------------------------------
    def encode_texts(self, texts: Sequence[str], batch_size: int = 1024) -> np.ndarray:
        torch = self._torch
        out: List[np.ndarray] = []
        # OpenCLIP tokenizer truncates to 77 tokens by default — fine for caption-like text.
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            tokens = self.tokenizer(chunk).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out.append(emb.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.DIM), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Sentence-Transformers (text only)
# ---------------------------------------------------------------------------


class SentenceEncoder:
    """all-MiniLM-L6-v2 — 384 dim, ~1500 sentences/sec on H100. Default choice
    for text-heavy datasets (SQuAD, TriviaQA, MS MARCO).
    """

    DIM = 384
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: Optional[str] = None, model_name: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        if model_name:
            self.MODEL_NAME = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.MODEL_NAME, device=self.device)
        self.DIM = int(self.model.get_sentence_embedding_dimension())

    def encode_texts(self, texts: Iterable[str], batch_size: int = 512) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)
