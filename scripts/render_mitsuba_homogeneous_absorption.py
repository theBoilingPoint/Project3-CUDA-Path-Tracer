#!/usr/bin/env python3
"""Render the homogeneous-absorption validation scene in Mitsuba 3.

This is a matched reference for
``scenes/jsons/volumetric/validation/homogeneous_absorption_cube.json``.
It intentionally keeps the scene description in Python so it can be run from
the Mitsuba package installed in the Conda base environment without generating
or maintaining a second scene file.

The EXR stores unexposed scene-linear radiance. The PNG follows the CUDA path
tracer's ``TONEMAP=false`` output convention: multiply by ``2**EXPOSURE``,
clamp to [0, 1], and quantize linearly to 8 bits (no sRGB transfer function).
The deliberately low exposure keeps both the unobscured panel and the
attenuated center below display clipping, which makes image metrics useful.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import time

import drjit as dr
import mitsuba as mi


VARIANT = "cuda_ad_rgb"
RESOLUTION = (320, 320)
FOV_Y_DEGREES = 28.0
SPP_DEFAULT = 32
SEED_DEFAULT = 20260804
EXPOSURE_STOPS = -3.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scene(spp: int) -> mi.Scene:
    """Build the geometry/material/camera-equivalent Mitsuba scene."""
    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {
                "type": "volpath",
                "max_depth": 8,
                # Keep Russian roulette outside the source scene's depth cap.
                "rr_depth": 9,
            },
            "sensor": {
                "type": "perspective",
                "fov": FOV_Y_DEGREES,
                "fov_axis": "y",
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[0.0, 0.0, 5.0],
                    target=[0.0, 0.0, 0.0],
                    up=[0.0, 1.0, 0.0],
                ),
                "sampler": {
                    "type": "independent",
                    "sample_count": spp,
                },
                "film": {
                    "type": "hdrfilm",
                    "width": RESOLUTION[0],
                    "height": RESOLUTION[1],
                    "pixel_format": "rgb",
                    "component_format": "float32",
                    "rfilter": {"type": "box"},
                },
            },
            # Both renderers' canonical cubes span [-1, 1]^3 here. A null
            # boundary changes only the current medium and is not visible.
            "absorbing_cube": {
                "type": "cube",
                "bsdf": {"type": "null"},
                "interior": {
                    "type": "homogeneous",
                    # Project coefficients: sigma_a=.5, sigma_s=0, density=1.
                    "sigma_t": {
                        "type": "rgb",
                        "value": [0.5, 0.5, 0.5],
                    },
                    "albedo": {
                        "type": "rgb",
                        "value": [0.0, 0.0, 0.0],
                    },
                },
            },
            # The source panel is a cube centered at z=-3 with z-scale .08;
            # the first emissive face seen by the camera is therefore z=-2.96.
            # Mitsuba's canonical rectangle spans [-1, 1]^2.
            "reference_panel": {
                "type": "rectangle",
                "to_world": (
                    mi.ScalarTransform4f.translate([0.0, 0.0, -2.96])
                    @ mi.ScalarTransform4f.scale([3.0, 3.0, 1.0])
                ),
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "rgb",
                        "value": [4.0, 4.0, 4.0],
                    },
                },
            },
        }
    )


def bitmap_float_view(bitmap: mi.Bitmap) -> ctypes.Array[ctypes.c_float]:
    """Expose a float32 Mitsuba bitmap without requiring NumPy."""
    if bitmap.component_format() != mi.Struct.Type.Float32:
        raise TypeError(f"expected float32 film, got {bitmap.component_format()}")
    count = bitmap.buffer_size() // ctypes.sizeof(ctypes.c_float)
    array_type = ctypes.c_float * count
    # ``Object.ptr`` is the Mitsuba C++ object address, not the pixel buffer.
    # The standard array interface publishes the actual host data address.
    data_address = bitmap.__array_interface__["data"][0]
    return ctypes.cast(data_address, ctypes.POINTER(array_type)).contents


def bitmap_bytes(bitmap: mi.Bitmap) -> bytes:
    data_address = bitmap.__array_interface__["data"][0]
    return ctypes.string_at(data_address, bitmap.buffer_size())


def write_outputs(
    linear: mi.Bitmap, exr_path: Path, png_path: Path
) -> tuple[str, float, float, float]:
    """Write unexposed EXR and the project's linear-clamped PNG equivalent."""
    linear.write(str(exr_path))

    # Work on a host-side copy so the EXR remains scene-linear and unexposed.
    exposed = mi.Bitmap(linear)
    exposure_scale = 2.0**EXPOSURE_STOPS
    exposed_values = bitmap_float_view(exposed)
    for index, value in enumerate(exposed_values):
        exposed_values[index] = min(max(value * exposure_scale, 0.0), 1.0)
    png = exposed.convert(
        mi.Bitmap.PixelFormat.RGB,
        mi.Struct.Type.UInt8,
        False,  # preserve the project's deliberately linear 8-bit encoding
    )
    png.write(str(png_path))
    values = bitmap_float_view(linear)
    return (
        hashlib.sha256(bitmap_bytes(linear)).hexdigest(),
        min(values),
        max(values),
        sum(values) / len(values),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/official_repo_audit"),
    )
    parser.add_argument("--spp", type=int, default=SPP_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--runs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.spp <= 0 or args.runs <= 0:
        raise ValueError("--spp and --runs must both be positive")

    mi.set_variant(VARIANT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    linear_hashes: list[str] = []
    scene = build_scene(args.spp)
    for run_index in range(1, args.runs + 1):
        start = time.perf_counter()
        mi.render(scene, spp=args.spp, seed=args.seed)
        dr.sync_thread()
        # Developing the film transfers the scene-linear result to the host.
        linear = scene.sensors()[0].film().bitmap()
        elapsed = time.perf_counter() - start

        stem = f"mitsuba_homogeneous_absorption_run{run_index}"
        exr_path = args.output_dir / f"{stem}.exr"
        png_path = args.output_dir / f"{stem}.png"
        linear_hash, linear_min, linear_max, linear_mean = write_outputs(
            linear, exr_path, png_path
        )
        linear_hashes.append(linear_hash)

        records.append(
            {
                "run": run_index,
                "seconds": elapsed,
                "linear_array_sha256": linear_hash,
                "exr": str(exr_path),
                "exr_sha256": sha256(exr_path),
                "png": str(png_path),
                "png_sha256": sha256(png_path),
                "linear_min": linear_min,
                "linear_max": linear_max,
                "linear_mean": linear_mean,
            }
        )

    arrays_exact = len(set(linear_hashes)) == 1
    exrs_exact = len({record["exr_sha256"] for record in records}) == 1
    pngs_exact = len({record["png_sha256"] for record in records}) == 1
    report = {
        "renderer": f"Mitsuba {mi.__version__}",
        "variant": VARIANT,
        "resolution": list(RESOLUTION),
        "spp": args.spp,
        "seed": args.seed,
        "runs": args.runs,
        "exposure_stops_png_only": EXPOSURE_STOPS,
        "exact_reproducibility": {
            "linear_arrays": arrays_exact,
            "exr_files": exrs_exact,
            "png_files": pngs_exact,
        },
        "analytic_center_radiance": 4.0 * math.exp(-0.5 * 2.0),
        "assumptions": [
            "The source and Mitsuba cubes both span [-1,1]^3.",
            "The thin emissive cube is represented by its camera-facing plane at z=-2.96.",
            "EMITTANCE=4 is mapped to a constant area-emitter radiance of 4.",
            "Both emitters are untinted RGB/D65 lights with radiance 4.",
            "The EXR is unexposed scene-linear RGB.",
            "The PNG uses exposure -3, clamp, and linear uint8 quantization without sRGB transfer.",
            "A box reconstruction filter avoids a cross-renderer filter mismatch.",
        ],
        "records": records,
    }
    report_path = args.output_dir / "mitsuba_homogeneous_absorption_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not (arrays_exact and exrs_exact and pngs_exact):
        raise RuntimeError("fixed-seed render was not exactly reproducible")


if __name__ == "__main__":
    main()
