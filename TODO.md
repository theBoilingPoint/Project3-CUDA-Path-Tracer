# Volumetric rendering diagnosis and showcase

Last updated: 2026-07-28

Status: complete

## Final verdict

The cotton-like result was caused by both implementation design and scene
tuning. It was not primarily a Henyey-Greenstein sign error or a broken
Woodcock tracker.

The decisive implementation issue was the density model:

- `cloudDensity` was a max-union of 10 solid spheres.
- `plumeDensity` was a max-union of 16 expanding spheres.
- Noise only eroded those primitive unions, so their silhouettes necessarily
  retained a cotton-ball scaffold.

The decisive tuning/display issues were:

- Original media had optical depths in the tens, making them behave like
  opaque solids.
- Linear radiance was clamped directly into the preview/PNG. In the supplied
  `volumetric_cloud_sky.png`, 64.5% of segmented cloud pixels had at least one
  clipped channel and 24.6% were pure white.
- Broad frontal/overhead illumination removed most readable extinction
  gradients.
- The original "fog" was a small, optically thick tinted sphere, so it read as
  a transparent brown object rather than atmospheric depth.

The revised implementation and scenes separate these concerns: cloud, fog,
smoke, and fire now use distinct density, extinction, phase, lighting, and
composition choices.

## Completed work

- [x] Audited medium free-flight sampling, null collisions, phase sampling,
  boundary handling, direct lighting, and shadow transmittance.
- [x] Replaced sphere-union cloud and smoke density fields.
- [x] Added a dedicated soft ground-fog profile.
- [x] Added a tapered/forked flame profile and participating-medium emission.
- [x] Added tone mapping, exposure, sRGB output, linear HDR output, and
  optional OIDN CUDA denoising.
- [x] Added deterministic headless rendering.
- [x] Added isolated cloud, fog, smoke, and fire scenes.
- [x] Added a combined scene with all four effects.
- [x] Rendered and visually inspected low-sample previews after each material,
  lighting, and composition pass.
- [x] Regenerate final showcase PNG, raw PNG, and HDR outputs after the last
  audited density/emission changes.
- [x] Regenerate the Nsight Systems profile against the final combined scene.
- [x] Rebuilt and validated all final scene JSON files.

## Original-scene measurements

- The initial sky cloud used `DENSITY = 6`, approximately
  `sigma_t = 1.003`, and a `15 x 6.5 x 7` container. Its peak mean free path
  was about 0.166 world units.
- The Cornell cloud had a peak mean free path near 0.110 world units.
- The smoke peak mean free paths were approximately 0.12 to 0.23 world units.
- The original amber-fog sphere had center optical depth approximately
  `[4.05, 4.86, 7.56]`, leaving only about
  `[1.7%, 0.78%, 0.052%]` direct transmission.

These values explain the opaque/cotton appearance even with otherwise correct
transport.

## Transport audit

Confirmed correct:

- HG evaluation and sampling use one consistent propagation-direction
  convention. Positive `G` is forward scattering.
- Heterogeneous free flight uses hero-wavelength delta tracking against a
  valid constant majorant.
- Heterogeneous shadow transmittance uses ratio tracking.
- Density profiles remain bounded in `[0, 1]`, so the majorant remains valid.
- Homogeneous media use analytic exponential free-flight sampling.
- Environment, area, point, and directional lights are sampled at real medium
  events through the existing NEE/MIS closure path.

Correctness fixes made during the audit:

- Null boundaries no longer consume the effective path-depth budget through
  the outer traversal loop.
- Area-light MIS now uses total distance since the last real vertex, including
  distance crossed before and after null interfaces.
- Signed-overflow undefined behavior was removed from the RNG/noise hashes.
- HG inverse sampling is clamped at extreme `G`.
- The camera is explicitly rejected when it starts inside a medium because the
  current path state assumes vacuum initially.
- Emissive-medium integration now uses an independent four-stratum,
  full-spectrum estimate of `integral T(0,s) j(s) ds`; it is not coupled to
  the hero-wavelength free-flight event.
- Terminated secondary spectral wavelengths contribute zero at the sensor.
- Scene loading rejects shaped profiles on homogeneous media and scans
  referenced geometry, rather than unused material declarations, when
  validating that a scene has illumination.
- PNG/HDR write failures propagate to a nonzero headless exit status.
- JSON `FOVY` is now interpreted as a full vertical field of view using
  `tan(FOVY / 2)`.
- The headless accumulation divisor is restored to the actual SPP count after
  the render loop.

## Density and appearance changes

### Cloud

- One connected, smooth-summed implicit condensation shelf replaces the
  former max-union of solid sphere primitives.
- Several anisotropic 3-D updraft fields merge through a saturating sum,
  giving the cloud a coherent base, shoulders, and rising towers without
  exposing the individual construction fields.
- Gaussian field tails provide natural tapering. A depth-coherent irregular
  base and silhouette erosion prevent a flat cutout or container edge.
- Three-dimensional value-noise/Worley modulation erodes the interior and
  creates smaller wisps without breaking the macro shape into cotton balls.
- Moderate optical depth and oblique lighting preserve core shading instead
  of clipping the entire body to white.

### Fog

- A low heterogeneous bank with broad horizontal variation, denser low
  layers, sparse lifted wisps, and a vertical fade before the container top.
- Three progressively smaller red markers at increasing depth make extinction
  immediately visible.
- The final view shows progressive desaturation/occlusion rather than a hard
  tinted sphere.

### Smoke

- One continuous rising column around a wandering advected axis.
- Radius expands gradually and dissipates at the top; there is no terminal
  mushroom sphere.
- Height-stretched noise produces sheets and wisps.
- Absorption dominates scattering, producing a dark soot core.
- Outdoor sunset backlighting makes the upper plume readable; a restrained
  warm source light colors only the lower plume.

### Fire

- A tapered main tongue plus two upper forks.
- Density and temperature vary spatially.
- Continuous blackbody-compatible medium emission is integrated along each
  traveled segment with four stratified samples.
- A colocated warm point-light proxy illuminates nearby exterior surfaces.

## Implementation map

- `src/render/volume.h`
  - HG helpers.
  - Safe deterministic value/Worley/fBm noise.
  - `Cloud`, `Fog`, `Plume`, and `Flame` density profiles.
  - Homogeneous/heterogeneous free-flight and transmittance estimators.
  - Spatial flame emission and relative temperature.
- `src/render/pathtrace.cu`
  - Medium integration and emission.
  - Null-interface traversal and MIS-distance bookkeeping.
  - Stable domain-separated RNG seeding.
  - Display transform for the interactive PBO.
- `src/core/color.h`
  - ACES fitted curve, exposure, and exact linear-to-sRGB transfer.
- `src/core/main.cpp`
  - `--headless`, `--spp`, `--output`, and `--denoise`.
  - Raw/denoised/HDR save workflow.
- `src/scene/scene.cpp`, `src/scene/sceneStructs.h`
  - Medium emission, profile, seed, validation, camera exposure/tone-map
    fields, and corrected FOV handling.
- `src/display/image.cpp`
  - Checked PNG/HDR writes.
  - Nonnegative scene-linear Radiance RGBE output; signed out-of-gamut and
    non-finite components are clipped because RGBE cannot represent them.

## Scene files

| Effect | Scene |
| --- | --- |
| Cloud | `scenes/jsons/volumetric/showcase_cloud.json` |
| Fog | `scenes/jsons/volumetric/showcase_fog.json` |
| Smoke | `scenes/jsons/volumetric/showcase_smoke.json` |
| Fire | `scenes/jsons/volumetric/showcase_fire.json` |
| Combined | `scenes/jsons/volumetric/showcase_combined.json` |

The combined scene keeps the four medium containers spatially disjoint. The
current path state stores only one active medium index and therefore does not
support nested or overlapping media.

## Reproducible commands

Build:

```powershell
cmake --build .\build --config Release --target cis565_path_tracer
```

Validate all showcase JSON:

```powershell
Get-ChildItem .\scenes\jsons\volumetric -Filter 'showcase_*.json' |
  ForEach-Object {
    Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json | Out-Null
  }
```

Render pattern:

```powershell
$spp = @{
  cloud = 1536
  fog = 1536
  smoke = 1536
  fire = 1024
  combined = 1536
}

$spp.GetEnumerator() | ForEach-Object {
  .\build\bin\Release\cis565_path_tracer.exe `
    ".\scenes\jsons\volumetric\showcase_$($_.Key).json" `
    --headless --denoise --spp $_.Value `
    --output ".\img\volumetric_$($_.Key)_showcase"
}
```

`--denoise` requests OIDN for the display PNG:

- `<base>.png` is OIDN CUDA denoised when OIDN succeeds, then
  exposure/tone-mapped/sRGB encoded; it falls back to the raw display if OIDN
  is unavailable or fails.
- `<base>.raw.png` is always written when `--denoise` is requested and contains
  the undenoised accumulated frame after the same display transform.
- `<base>.hdr` is nonnegative scene-linear radiance. Negative/non-finite
  out-of-gamut components are clipped for Radiance RGBE storage.

## Final renders

| Effect | Resolution | SPP | Time | Display PNG |
| --- | ---: | ---: | ---: | --- |
| Cloud | 800 x 450 | 1536 | 71.86 s | `img/volumetric_cloud_showcase.png` |
| Fog | 800 x 450 | 1536 | 42.36 s | `img/volumetric_fog_showcase.png` |
| Smoke | 800 x 450 | 1536 | 51.28 s | `img/volumetric_smoke_showcase.png` |
| Fire | 800 x 450 | 1024 | 42.04 s | `img/volumetric_fire_showcase.png` |
| Combined | 960 x 540 | 1536 | 134.92 s | `img/volumetric_combined_showcase.png` |

Each display PNG has matching `.raw.png` and `.hdr` files in `img/`.

Visual acceptance notes:

- Cloud: connected low cloud bank, irregular condensation base, asymmetric
  updraft towers, self-shadowed core, and no discrete sphere chain.
- Fog: soft bank with clearly progressive marker loss over depth and no hard
  container ceiling.
- Smoke: dark, continuous S-shaped plume with warm dense source and cool
  dissipating top.
- Fire: bright hot core, cooler tapered/forked tip, visible environmental glow,
  and no emissive surface standing in for the flame volume.
- Combined: all four regimes remain visually distinct in one frame.

## Nsight Systems result

Tool:

```text
NVIDIA Nsight Systems 2026.1.3
```

`nsys` was not on this PowerShell session's `PATH`; the executable resolved to
`C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe`.

Capture command:

```powershell
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe'
& $nsys profile --trace=cuda --sample=none --cpuctxsw=none `
  --force-overwrite=true `
  --output=.\.cache\volumetric_combined_nsys `
  .\build\bin\Release\cis565_path_tracer.exe `
  .\scenes\jsons\volumetric\showcase_combined.json `
  --headless --spp 24 --output .\.cache\nsys_combined_render
```

Report:

```text
.cache/volumetric_combined_nsys.nsys-rep
```

CUDA GPU time breakdown:

- `shade`: 68.7%
- CUB merge-sort merge kernel: 20.5%
- `computeIntersections`: 4.4%
- CUB block sort: 3.0%
- Remaining kernels: 3.4%

CUDA API time is dominated by `cudaDeviceSynchronize` (55.6%) and
`cudaStreamSynchronize` (29.3%). This profile identifies synchronization and
per-bounce material sorting as the main performance opportunities; it does not
indicate a volumetric correctness failure. `shade` includes the procedural
density, free-flight, transmittance, phase, and medium-emission work, so its
larger final share is expected after replacing the primitive density fields.

## Validation result

- Release build succeeds.
- All five final scene files parse as JSON.
- Headless rendering, raw HDR output, raw PNG output, and OIDN CUDA output all
  execute successfully.
- A strict raw/denoised visual review passes the standalone cloud for the
  sphere-chain/cotton criterion and passes all four effects in the combined
  frame.
- `ctest --test-dir .\build -C Release --output-on-failure` succeeds, but this
  repository currently registers no automated tests.
- `git diff --check` reports no whitespace errors (only the repository's
  existing LF-to-CRLF conversion warnings).
- Expected compiler/linker warnings remain:
  - existing GLM CUDA annotation warnings,
  - existing `APIENTRY` macro redefinition warning,
  - existing `LNK4098` runtime-library warning.

## Known limitations and follow-up

- Nested/overlapping media are unsupported because each path stores one active
  medium index. A future implementation should use a medium stack or priority
  system.
- Cameras inside media are unsupported and rejected at load time.
- Emissive media are visible to camera and indirect paths, but are not sampled
  as extended lights from arbitrary exterior points. The fire scenes use a
  warm point proxy for exterior illumination.
- Color-only OIDN can smooth very fine low-SNR volume structure. Raw PNG and
  linear HDR are always retained for evaluation.
- Nsight shows that removing unconditional synchronization and reconsidering
  full material sorting would improve performance more than micro-optimizing
  the procedural noise.
