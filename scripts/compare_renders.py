#!/usr/bin/env python3
"""Compare two aligned deterministic renders.

PSNR and SSIM are evaluated in display RGB [0, 1]. LPIPS is optional because
it depends on PyTorch and the external ``lpips`` package. The script rejects
different resolutions: resampling would make implementation comparisons look
better than they are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_lpips(reference: np.ndarray, candidate: np.ndarray,
                  device: str) -> float:
    try:
        import lpips
        import torch
    except ImportError as error:
        raise RuntimeError(
            "LPIPS requires the optional torch and lpips packages") from error

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = lpips.LPIPS(net="alex").to(device).eval()

    def tensor(image: np.ndarray):
        value = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return (value * 2.0 - 1.0).to(device)

    with torch.inference_mode():
        return float(metric(tensor(reference), tensor(candidate)).item())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PSNR/SSIM/LPIPS for pixel-aligned deterministic PNGs")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path,
                        help="optional JSON output path")
    parser.add_argument("--lpips", action="store_true",
                        help="also compute AlexNet LPIPS")
    parser.add_argument("--lpips-device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args()

    reference = load_rgb(args.reference)
    candidate = load_rgb(args.candidate)
    if reference.shape != candidate.shape:
        raise SystemExit(
            f"images must be aligned and equally sized: "
            f"{reference.shape} != {candidate.shape}")

    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise SystemExit(
            "SSIM requires scikit-image; install the metric dependencies "
            "documented in README.md") from error

    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(difference * difference))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    ssim = float(structural_similarity(
        reference, candidate, channel_axis=2, data_range=1.0))

    result: dict[str, object] = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "resolution": [int(reference.shape[1]), int(reference.shape[0])],
        "reference_sha256": file_sha256(args.reference),
        "candidate_sha256": file_sha256(args.candidate),
        "exact_file_match": file_sha256(args.reference) ==
                            file_sha256(args.candidate),
        "mse_display_rgb": mse,
        "psnr_display_rgb_db": psnr,
        "ssim_display_rgb": ssim,
    }
    if args.lpips:
        result["lpips_alex_display_rgb"] = compute_lpips(
            reference, candidate, args.lpips_device)

    encoded = json.dumps(result, indent=2, allow_nan=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
