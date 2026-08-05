# Volumetric Rendering Status

Last updated: 2026-08-04

Status: in progress. The earlier fire showcase and smoke/soot/fog transport
audit remain frozen; two new Cornell-box recreation targets are being rendered
in Blender, PBRT, Mitsuba, and this project.

## Cornell Fire/Soot and Diagonal-Smoke Recreations

- [x] Convert the supplied images into renderer-independent shape, opacity,
  lighting, and color-transition acceptance criteria rather than attempting a
  misleading pixel match.
- [x] Start aligned 512 x 512 static heterogeneous-volume references in
  Blender/Cycles, PBRT v4, and Mitsuba 3 with raw HDR/EXR retained.
- [x] Add `cornell_diagonal_smoke_plume.json`: a rotated, coherent 3-D plume
  with a dense warm-black core, broad translucent head, HG scattering, and a
  red/green Cornell enclosure.
- [x] Add `cornell_fire_soot_cloud.json`: one sparse field containing density,
  soot, fuel, reaction, temperature, and velocity; no billboard or overlapping
  volume is used.
- [x] Add opt-in compact-source, canopy-spread, and baked soot-canopy controls
  to let a small burner feed a connected broad smoke mass while preserving the
  existing WoodFire defaults.
- [ ] Accept final project macro-shape, fire white/yellow/orange transition,
  smoke density gradient, and room illumination at useful SNR.
- [ ] Finish and inspect all six official-reference renders.
- [ ] Publish eight individual images and two labeled side-by-side comparison
  sheets, then record settings, timings, and hashes.
- [ ] Rebuild, run CTest, validate scene JSON, and refresh README status and
  reproduction commands.

## Final Delivery Checklist

- [x] Validate and freeze the `192 x 160 x 160` wildfire grid after the
  high-quality beauty gate passed.
- [x] Render raw low/medium-sample wildfire gates and reject them when fixed-root
  pickets, horizontal shelves, detached highlights, or a uniform yellow mass
  returned.
- [x] Freeze and render post-port candle, wood-fire, and wildfire scenes.
- [x] Retain display PNG, non-denoised `.raw.png`, and scene-linear `.hdr` for
  every final beauty.
- [x] Review raw and denoised output together. OIDN does not invent flame
  connectivity or hide unresolved grid structure.
- [x] Rebuild Release and rerun the registered CTest target plus all ten
  controlled volumetric scenes after the final scene/code freeze.
- [x] Capture a final Nsight Systems profile of the accepted 192-grid wildfire
  after its beauty render is accepted.
- [x] Fill the final table below with measured SPP, timings, hashes, and
  acceptance notes.
- [x] Upload the accepted artifacts to the requested
  [Google Drive folder](https://drive.google.com/drive/folders/18TMgF1j5UxUOooPnN2XOiyb-8whefwUB)
  and verify names, sizes, MIME types, and parent-folder metadata.
- [x] Render the same 320 x 320 homogeneous soot, fog, and smoke scenes at
  4096 spp in the project, Blender/Cycles, PBRT v4, and Mitsuba 3.
- [x] Normalize every raw HDR/EXR through the same `imgtool --scale 0.25`
  display conversion before PSNR, SSIM, or LPIPS evaluation.
- [x] Establish repeatability before using metrics: project and PBRT pixels
  are exact; Cycles and Mitsuba are numerically stable at negligible
  parallel-reduction error.
- [x] Replace stochastic homogeneous pure absorption with a variance-free
  Beer-Lambert GPU fast path and rerun tests, the soot reference, and Nsight
  Systems.
- [x] Upload and metadata-verify all 30 files in the official reference
  [comparison subfolder](https://drive.google.com/drive/folders/1lPvs7WDBx4DI2KgV1qbjjme13Z943Xg4).

## Final Beauty Publication Gate

| Scene | Resolution | SPP | Time | Raw QA | Display QA | Drive |
|:--|:--:|--:|--:|:--:|:--:|:--:|
| Candle | `800 x 600`; grid `48 x 96 x 48` | 1024 | 109.85 s | Accepted | Accepted | Published |
| Wood fire | `960 x 540`; grid `112 x 128 x 80` | 1024 | 308.95 s | Accepted | Accepted | Published |
| Wildfire | `960 x 540`; grid `192 x 160 x 160` | 512 | 264.87 s | Accepted | Accepted | Published |

All times are renderer-reported trace times; scene loading and static field
generation are outside that timer. Exact artifact sizes and SHA-256 hashes are
in `img/volumetric_fire_showcases_manifest.json`.
Drive readback confirmed ten files, matching local byte sizes, expected PNG,
Radiance HDR, and JSON MIME types, and the requested folder as the parent.

## Completed Renderer Work

### Static combustion fields

- [x] Added deterministic static semi-Lagrangian field generation at scene
  load.
- [x] Stored density, soot, fuel, reaction, temperature, and velocity.
- [x] Added separate candle, wood-fire, and wildfire source presets.
- [x] Used wind, buoyancy, and multiscale divergence-free curl noise for
  coherent field advection.
- [x] Kept smoke, soot, fuel, and temperature cooling/dissipation independent.
- [x] Coupled char, ash, coal, burned ground, vegetation, and static emissive
  embers to the corresponding showcase compositions.
- [x] Removed billboard, crossed-quad, scrolling-alpha, and polygonal main
  flame representations.

### Sparse GPU storage

- [x] Added sparse `8^3` logical bricks with duplicated `9^3` interpolation
  halos.
- [x] Added a dense brick page table, compact active-brick payloads, and brick
  DDA empty-space skipping.
- [x] Added conservative halo maxima with order-independent 26-neighbor
  expansion and outward rounding.
- [x] Preserved the medium object's world-to-volume transform.
- [x] Packed active payloads into two float4 CUDA 3-D texture atlases.
- [x] Precomputed every brick's atlas origin and replaced manual trilinear
  loads with two hardware-filtered `tex3D` operations.
- [x] Kept page-table traversal and local majorants independent of the texture
  representation.

### Optical model and transport

- [x] Derived absorption, scattering, extinction, and emission independently
  from stored matter/temperature fields.
- [x] Added analytic Planck blackbody emission plus temperature-gated
  incandescent-soot emission in unclamped HDR.
- [x] Fixed equal-energy illuminant double-normalization discovered by the
  matched Mitsuba comparison.
- [x] Added local-majorant delta tracking for camera/continuation free flight.
- [x] Added local-majorant ratio tracking for shadow transmittance.
- [x] Added Henyey-Greenstein phase sampling, multiple volume scattering,
  volume next-event estimation, and power-heuristic MIS.
- [x] Added an analytic Beer-Lambert continuation path when homogeneous
  `sigma_s` is zero, eliminating a useless absorption/escape Bernoulli event.
- [x] Added ordered stratified segment emission with independent transmittance
  tracking.
- [x] Added a brick/cell emitted-power hierarchy and a bounded local
  inverse-square mixture proposal for vertices embedded in fire.
- [x] Added explicit extended-volume-emitter sampling from surface and medium
  vertices, allowing fire to illuminate nearby geometry and smoke.
- [x] Added a separate PBRT-style power-weighted surface-light CDF.
- [x] Used the same surface-light PMF in NEE and hit-light MIS.
- [x] Corrected nonuniform affine-sphere light sampling with the exact
  local-to-world area Jacobian.

### Diagnostics and output

- [x] Added reference/debug quality modes consuming the same fields.
- [x] Added temperature, density, fuel, soot, reaction, emission,
  absorption/scattering/extinction, velocity, majorant, collision, event, and
  contribution AOVs.
- [x] Added optional volume statistics without allocating counters when
  disabled.
- [x] Added deterministic headless rendering, SPP/output overrides, raw PNG,
  scene-linear HDR, exposure, optional OIDN CUDA denoising, and a conditional
  display path: ACES plus sRGB when `TONEMAP` is enabled, or exposed, clamped
  linear RGB when it is disabled.

## Official Renderer Audit

Local source revisions:

| Renderer | Revision/runtime | Findings and adopted work |
|:--|:--|:--|
| Blender/Cycles | Blender `d769b0ee1e3`; Blender CLI 5.1.2, OptiX | Audited volume stacks, octree majorants, null scattering, equiangular/distance MIS, blackbody handling, and GPU volume integration. Its analytic absorption behavior motivated the new fast path; guiding and equiangular sampling remain profile-gated future work. |
| Mitsuba 3 | `5f090a15`; Mitsuba 3.9.0, Dr.Jit 1.4.0, `cuda_ad_rgb` | Audited CUDA texture interpolation, homogeneous and heterogeneous media, HG, volume NEE/MIS, and null visibility. The project already has these transport features plus sparse local majorants. |
| PBRT v4 | `7154d82`; recursive CPU `volpath` and `imgtool` | Audited spectral coefficients, HG, NanoVDB/grid media, DDA majorants, null collisions, ratio tracking, and power-light sampling. No correctness port was missing; NanoVDB import remains optional interoperability work. |

License boundary:

- Blender application source is GPL.
- Audited Cycles files under `intern/cycles` are individually Apache-2.0.
- PBRT v4 is Apache-2.0.
- Mitsuba 3 and Dr.Jit are BSD-3-Clause.
- No official-renderer implementation was copied wholesale; adapted concepts
  are documented above.

## Validation and Determinism

### Project tests

- [x] Registered one CTest target containing eleven internal sparse-volume
  groups.
- [x] Covered preset construction, exact trilinear interpolation, page-table
  invariants, conservative neighbor majorants, brick/cell CDF and PDF support,
  sampled/evaluated PDF agreement, Beer-Lambert attenuation, the
  absorption-only free-flight fast path, coefficient identities,
  Henyey-Greenstein normalization/moment/sampling, Planck behavior, and
  malformed-resolution rejection.
- [x] Added thirteen controlled scenes:
  `homogeneous_absorption_cube`, `homogeneous_emitting_sphere`,
  `homogeneous_scattering_point`, `heterogeneous_known_majorant`,
  `blackbody_temperature_ramp`, `backlit_smoke`, `single_burning_log`,
  `three_log_campfire`, `grass_strip_wind`, `wildfire_grid_fields`,
  `reference_soot_absorption`, `reference_fog_scattering`, and
  `reference_smoke_scattering`.
- [x] Verified fixed-seed reproducibility: repeated absorbing and stochastic
  scattering renders produced byte-identical PNG/HDR hashes.

### Aligned smoke, soot, and fog metrics

All four renderers use the same 320 x 320 camera, `[-1,1]^3` medium, 6 x 6
backlight with scene-linear radiance 4, black world, depth 12, seed 20260804,
4096 spp, and physical coefficients. Raw HDR/EXR files are retained; the
metric PNGs all use PBRT `imgtool convert --scale 0.25` with no denoising.
Metrics are implementation comparisons, not realism scores.

| Medium | Reference | PSNR | SSIM | LPIPS (AlexNet) |
|:--|:--|--:|--:|--:|
| Soot | Blender | 46.3783 dB | 0.994043 | 0.003233 |
| Soot | PBRT | 42.2976 dB | 0.958151 | 0.158097 |
| Soot | Mitsuba | 42.6118 dB | 0.964107 | 0.157157 |
| Fog | Blender | 48.2466 dB | 0.984537 | 0.006493 |
| Fog | PBRT | 46.2880 dB | 0.975498 | 0.000910 |
| Fog | Mitsuba | 47.3968 dB | 0.980979 | 0.002447 |
| Smoke | Blender | 44.4404 dB | 0.971469 | 0.101108 |
| Smoke | PBRT | 42.0947 dB | 0.947452 | 0.007550 |
| Smoke | Mitsuba | 42.6334 dB | 0.953728 | 0.008778 |

The results are comparable to the official-renderer spread: official pairwise
PSNR ranges are 41.22-44.63 dB for soot, 47.88-51.23 dB for fog, and
43.25-46.65 dB for smoke. The weakest project fog and smoke pairs are 1.60 dB
and 1.15 dB below those official minima, with no systematic shape or
attenuation mismatch. Fixed-seed project PNG/HDR pairs are byte exact; PBRT
repeat pixels are exact. Cycles repeat PSNR is at least 102.48 dB and Mitsuba
raw repeat PSNR is at least 143.78 dB, so their remaining variation is
floating-point reduction order rather than a changed sample sequence.

Evidence is under `.cache/reference_validation/`; the aggregate comparison
image, raw references, metrics, audits, and manifest are included in the Drive
delivery. Drive readback matched all 30 names, MIME types, parent IDs, and
12,962,727 local bytes.

## Nsight Evidence

Pre-grid and intermediate reports remain useful historical baselines, but only
the post-port capture describes the hardware-texture implementation.

- Texture-atlas origin precomputation moved the temporary low-resolution
  wildfire's 16-spp wall time from 8.57 s to 8.27 s.
- Final post-port report:
  `.cache/official_repo_audit/nsys_wildfire_final_192grid_4spp.nsys-rep`.
- Accepted 192-grid scene at 4 spp:
  - `shade`: 1.748 s, 81.3%.
  - `computeIntersections`: 0.244 s, 11.3%.
  - CUB merge: 5.9%.
  - CUB block sort: 0.8%.
  - Compaction: 0.3%.
- The homogeneous soot fast path reduced the same 320 x 320, 4096-spp render
  from 160.83 s to 63.40 s (60.6%) while removing absorption-event variance.
- In the post-port 64-spp Systems trace, `shade` is only 2.7% of GPU time;
  CUB merge sorting dominates at 66.6%, so further volume-shader ports are not
  justified by this workload.
- Post-port report:
  `.cache/reference_validation/project/nsys_soot_absorption_fastpath_64spp.nsys-rep`.
- Nsight Compute was attempted, but hardware counters are blocked by driver
  policy (`ERR_NVGPUCTRPERM`). No `.ncu-rep` is claimed.

## Diagnosis Summary

The initial cotton-like media came from both implementation design and tuning:

- Clouds used a max-union of ten solid spheres.
- Smoke used a max-union of sixteen expanding spheres.
- Noise eroded those primitives but retained their round silhouettes.
- Original media had optical depths in the tens and behaved like opaque solids.
- Direct linear-to-PNG clipping erased internal HDR gradients.
- Broad frontal/overhead light removed readable self-shadowing.
- The original fog was a small optically thick tinted sphere rather than an
  atmospheric depth cue.

The transport audit did not find an HG sign error or invalid Woodcock
majorant. The decisive changes were connected 3-D fields, moderate optical
depth, physically separated coefficients/emission, better lighting/exposure,
and sparse stored combustion fields.

## Static Scope and Known Limitations

- The requested output is static offline still imagery. Animation, grid-time
  interpolation, motion blur, dynamic fuel propagation, and animated ember
  trajectories/collisions are intentionally out of scope.
- There is no OpenVDB/NanoVDB importer; the project uses a custom static
  procedural sparse grid.
- Gradient-index heat refraction is not implemented.
- Nested or overlapping media are unsupported because a path stores one active
  medium index.
- Cameras starting inside media are rejected at load time.
- Embers are static authored emissive geometry, not a runtime particle system.
- Supplemental key/fill lights remain allowed, but fire emission participates
  in path transport and is not replaced by a point-light proxy.
- OIDN is color-only and may smooth low-SNR fine flame structure. Raw PNG and
  HDR are mandatory acceptance evidence.
- Wood and wildfire context meshes are purpose-built low-poly props rather than
  photographic vegetation.

## Reproducible Commands

Build and test:

```powershell
.\scripts\build.bat
cmake --build .\build --config Release --target volume_grid_tests
ctest --test-dir .\build -C Release -R volume_grid_tests --output-on-failure
```

Quick controlled-scene sweep:

```powershell
$exe = ".\build\bin\Release\cis565_path_tracer.exe"
$tests = @(
  "homogeneous_absorption_cube", "homogeneous_emitting_sphere",
  "homogeneous_scattering_point", "heterogeneous_known_majorant",
  "blackbody_temperature_ramp", "backlit_smoke",
  "single_burning_log", "three_log_campfire",
  "grass_strip_wind", "wildfire_grid_fields"
)
foreach ($name in $tests) {
  & $exe ".\scenes\jsons\volumetric\validation\$name.json" `
    --headless --spp 1 --output ".\.cache\validation\$name"
}
```

Deterministic comparisons:

```powershell
python .\scripts\compare_renders.py reference.png candidate.png --lpips
conda run -n base python .\scripts\render_mitsuba_homogeneous_absorption.py `
  --output-dir .\.cache\official_repo_audit\mitsuba_reference `
  --spp 4096 --seed 20260804 --runs 2
```

Beauty-render template (replace `<accepted-spp>` after the final gate):

```powershell
$exe = ".\build\bin\Release\cis565_path_tracer.exe"
& $exe .\scenes\jsons\volumetric\showcase_wildfire.json `
  --headless --denoise --spp <accepted-spp> `
  --output .\img\volumetric_wildfire_showcase
```

Final SPP and timing values remain placeholders until the publication gate is
complete.
