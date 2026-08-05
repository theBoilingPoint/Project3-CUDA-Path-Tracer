#pragma once

#include <cfloat>
#include <glm/glm.hpp>
#include <thrust/random.h>

#include "intersections.h" // multiplyMV, utilhash
#include "sceneStructs.h"  // Spectrum (via spectral.h), Geom, Material, Ray
#include "volumeGridDevice.h"

// ============================================================================
// Participating media: distance sampling and transmittance.
//
// Follows PBRT-v4's VolPathIntegrator (the null-scattering path integral of
// Miller, Georgiev & Jarosz 2019):
//   - Homogeneous media: analytic exponential distance sampling, with the
//     hero wavelength driving the sample and the Wilkie et al. 2014 balance
//     heuristic over the carried wavelengths' pdfs keeping the spectral
//     estimate unbiased.
//   - Heterogeneous media: delta (Woodcock) tracking against a constant
//     majorant for distance sampling, with per-wavelength null-collision
//     ratio weights (PBRT-v4's T_maj bookkeeping, collapsed to the
//     hero-driven single-sample form).
//   - Shadow rays: analytic transmittance (homogeneous) or ratio tracking
//     (Novak et al. 2014) for heterogeneous media.
//
// Density for heterogeneous media is a procedural fbm value-noise field in
// the geom's local unit space, normalized to [0, 1]; sigma arrays passed in
// are already uplifted to Spectrum and pre-scaled by Material::densityScale,
// so maxComponent(sigT) is a valid majorant.
//
// The Henyey-Greenstein phase function below uses the PROPAGATION-direction
// convention: cosTheta = dot(incomingDir, outgoingDir), so g > 0 scatters
// forward (clouds, smoke). The BSDF closure's phase lobe (bxdf.cu) converts
// from its wo/wi convention with cosTheta = dot(-woW, wiW).
// ============================================================================

// --- Henyey-Greenstein phase function ---------------------------------------

// Value == solid-angle pdf (the phase function is its own perfect importance
// sampler).
__host__ __device__ inline float phaseHG(float cosTheta, float g) {
    const float inv4Pi = 0.07957747154594767f;
    float denom = 1.0f + g * g - 2.0f * g * cosTheta;
    denom = fmaxf(denom, 1e-7f);
    return inv4Pi * (1.0f - g * g) / (denom * sqrtf(denom));
}

// Sample cosTheta (between incoming and outgoing propagation directions)
// proportional to phaseHG.
__host__ __device__ inline float sampleHGCosTheta(float g, float u) {
    if (fabsf(g) < 1e-3f) {
        return 1.0f - 2.0f * u; // isotropic limit
    }
    float s = (1.0f - g * g) / (1.0f + g - 2.0f * g * u);
    return glm::clamp((1.0f + g * g - s * s) / (2.0f * g), -1.0f,
                      1.0f);
}

// --- Procedural density field (heterogeneous media) -------------------------

// Deterministic lattice hash -> [0, 1].
__host__ __device__ inline float noiseHashUnsigned(unsigned int x,
                                                   unsigned int y,
                                                   unsigned int z) {
    unsigned int h =
        x * 73856093u ^ y * 19349663u ^ z * 83492791u;
    return (float)(utilhash(h) & 0x00FFFFFFu) / 16777215.0f;
}

__host__ __device__ inline float noiseHash(int x, int y, int z) {
    // Convert before multiplication so negative lattice coordinates wrap in
    // well-defined unsigned arithmetic instead of invoking signed-overflow UB.
    return noiseHashUnsigned((unsigned int)x, (unsigned int)y,
                             (unsigned int)z);
}

__host__ __device__ inline glm::vec3 noiseSeedOffset(int seed) {
    // Add the arbitrary decorrelation constants in unsigned arithmetic too:
    // JSON accepts the full int range, so `seed + constant` must not overflow.
    unsigned int s = (unsigned int)seed;
    return glm::vec3(noiseHashUnsigned(s + 17u, 31u, 47u),
                     noiseHashUnsigned(s + 59u, 71u, 89u),
                     noiseHashUnsigned(s + 97u, 101u, 131u)) *
           53.0f;
}

// 3D value noise: trilinear interpolation of lattice hashes with a smoothstep
// fade, result in [0, 1].
__host__ __device__ inline float valueNoise(const glm::vec3 &p) {
    glm::vec3 pf = glm::floor(p);
    int xi = (int)pf.x, yi = (int)pf.y, zi = (int)pf.z;
    glm::vec3 f = p - pf;
    glm::vec3 w = f * f * (3.0f - 2.0f * f);

    float c000 = noiseHash(xi, yi, zi);
    float c100 = noiseHash(xi + 1, yi, zi);
    float c010 = noiseHash(xi, yi + 1, zi);
    float c110 = noiseHash(xi + 1, yi + 1, zi);
    float c001 = noiseHash(xi, yi, zi + 1);
    float c101 = noiseHash(xi + 1, yi, zi + 1);
    float c011 = noiseHash(xi, yi + 1, zi + 1);
    float c111 = noiseHash(xi + 1, yi + 1, zi + 1);

    float x00 = c000 + w.x * (c100 - c000);
    float x10 = c010 + w.x * (c110 - c010);
    float x01 = c001 + w.x * (c101 - c001);
    float x11 = c011 + w.x * (c111 - c011);
    float y0 = x00 + w.y * (x10 - x00);
    float y1 = x01 + w.y * (x11 - x01);
    return y0 + w.z * (y1 - y0);
}

// Fractional Brownian motion over valueNoise, normalized to [0, 1].
__host__ __device__ inline float fbm(glm::vec3 p, int octaves) {
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * valueNoise(p);
        norm += amp;
        amp *= 0.5f;
        p *= 2.02f;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

// Worley (cellular) noise: distance to the nearest jittered feature point,
// in [0, 1] (0 at a feature point). Its INVERSE (1 - worley) forms packed
// bubble cells -- the cauliflower-billow ingredient that smooth value noise
// cannot produce (cf. Schneider's Horizon Zero Dawn cloudscapes).
__host__ __device__ inline float worley(const glm::vec3 &p) {
    glm::vec3 pf = glm::floor(p);
    int xi = (int)pf.x, yi = (int)pf.y, zi = (int)pf.z;
    glm::vec3 f = p - pf;
    float dmin = 1e9f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                glm::vec3 feature(
                    (float)dx + noiseHash(xi + dx, yi + dy, zi + dz),
                    (float)dy + noiseHash(yi + dy, zi + dz, xi + dx + 57),
                    (float)dz + noiseHash(zi + dz, xi + dx, yi + dy + 113));
                glm::vec3 d = feature - f;
                dmin = fminf(dmin, glm::dot(d, d));
            }
        }
    }
    return fminf(sqrtf(dmin), 1.0f);
}

// Three-octave inverted-Worley "billow" field in [0, 1]: 1 inside puffy
// cells, dipping toward 0 at cell borders -- carves cauliflower florets.
__host__ __device__ inline float billowFbm(const glm::vec3 &p) {
    return (1.0f - worley(p)) * 0.625f + (1.0f - worley(p * 2.03f)) * 0.25f +
           (1.0f - worley(p * 4.01f)) * 0.125f;
}

// --- Shaped density profiles ------------------------------------------------
// Real clouds and smoke are not noise filling a box: they have a large-scale
// SHAPE that high-frequency noise only decorates. Each profile builds a base
// mass in the geom's local unit space and erodes/breaks it with domain-warped
// fbm, the standard game/film procedural-volume recipe (cf. the "Nubis"
// cloudscapes talks and production VDB assets these emulate).

// A smooth anisotropic updraft field used by the cumulus macro shape. Several
// heavily overlapping fields are summed and saturated below; unlike a
// max-union of spheres, no individual primitive boundary survives.
__host__ __device__ inline float
cloudUpdraftField(const glm::vec3 &p, const glm::vec3 &center,
                  const glm::vec3 &radius, float weight) {
    glm::vec3 d = (p - center) / radius;
    return weight * expf(-2.25f * glm::dot(d, d));
}

// Cumulus profile: a genuinely three-dimensional implicit mass built from
// smooth-summed anisotropic updrafts, then carved by domain-warped
// value-noise/Worley detail. This avoids both the old cotton-ball max union
// and the projected cutout of a single-valued top height field.
__host__ __device__ inline float cloudDensity(const glm::vec3 &pl,
                                              float noiseScale, int octaves,
                                              int seed) {
    glm::vec3 seedP = noiseSeedOffset(seed);

    // Low-frequency 3-D warp destroys the analytic axes of the updraft fields
    // without breaking their connected mass into independent puffs.
    glm::vec3 macroQ = pl * glm::vec3(1.75f, 1.35f, 1.85f) + seedP;
    glm::vec3 macroWarp(
        fbm(macroQ * 0.70f + glm::vec3(11.3f, 3.7f, 19.1f), 3),
        fbm(macroQ * 0.70f + glm::vec3(29.7f, 7.1f, 5.3f), 3),
        fbm(macroQ * 0.70f + glm::vec3(2.9f, 37.7f, 13.9f), 3));
    glm::vec3 p =
        pl + (macroWarp - glm::vec3(0.5f)) *
                 glm::vec3(0.10f, 0.065f, 0.10f);

    // Broad condensation shelf, tall core, two shoulders, and a rear depth
    // mass. Their strong overlap makes one coherent body; the saturating sum
    // below is smooth everywhere and never selects a winning primitive.
    float sum = 0.0f;
    sum += cloudUpdraftField(p, glm::vec3(0.00f, -0.21f, 0.02f),
                            glm::vec3(0.43f, 0.13f, 0.35f), 0.95f);
    sum += cloudUpdraftField(p, glm::vec3(-0.22f, -0.06f, -0.04f),
                            glm::vec3(0.22f, 0.34f, 0.24f), 1.45f);
    sum += cloudUpdraftField(p, glm::vec3(0.00f, -0.02f, 0.07f),
                            glm::vec3(0.24f, 0.38f, 0.27f), 1.65f);
    sum += cloudUpdraftField(p, glm::vec3(0.22f, -0.08f, -0.05f),
                            glm::vec3(0.22f, 0.30f, 0.24f), 1.35f);
    sum += cloudUpdraftField(p, glm::vec3(-0.02f, -0.10f, -0.20f),
                            glm::vec3(0.31f, 0.24f, 0.18f), 0.75f);

    float weather =
        fbm(glm::vec3(p.x * 2.2f, p.z * 1.8f, 9.1f) + seedP * 0.17f, 4);
    sum *= 0.82f + 0.34f * weather;
    float mass = 1.0f - expf(-sum);

    // A physically plausible but visibly irregular condensation base. The
    // broad and detail bands produce approximately 0.4 world units of
    // variation at the showcase's current Y scale instead of a ruler line.
    // Keep the condensation-height variation coherent through depth so it
    // remains visible in projection instead of averaging back into a ruler.
    float baseLow =
        fbm(glm::vec3(p.x * 2.7f + seedP.x * 0.23f,
                      5.3f + seedP.y * 0.17f,
                      13.1f + seedP.z * 0.11f),
            3);
    float baseDetail =
        fbm(glm::vec3(p.x * 7.1f + seedP.x * 0.31f,
                      19.7f + seedP.y * 0.13f,
                      37.3f + seedP.z * 0.07f),
            3);
    float baseHeight = -0.31f + 0.09f * (baseLow - 0.5f) +
                       0.045f * (baseDetail - 0.5f);
    mass *= glm::smoothstep(baseHeight - 0.055f, baseHeight + 0.055f, p.y);

    // Production-style domain-warped erosion. It grows stronger near
    // low-density sides and upper levels, tapering the silhouette and opening
    // internal pockets while preserving a denser core.
    glm::vec3 q = p * noiseScale + seedP;
    glm::vec3 detailWarp(valueNoise(q * 0.35f + glm::vec3(13.1f)),
                         valueNoise(q * 0.35f + glm::vec3(47.7f)),
                         valueNoise(q * 0.35f + glm::vec3(91.3f)));
    float billow =
        billowFbm(q * 0.42f +
                  0.75f * (detailWarp - glm::vec3(0.5f)));
    float detail = fbm(q * 1.35f + glm::vec3(17.0f), octaves);
    float height = glm::clamp((p.y + 0.30f) / 0.70f, 0.0f, 1.0f);
    float erosionStrength =
        0.16f + 0.30f * (1.0f - mass) + 0.10f * height;
    float erosion = erosionStrength * (1.0f - billow) +
                    0.10f * (1.0f - detail);

    // A separate x/y band is deliberately coherent through depth. Fully 3-D
    // detail is physically useful inside the cloud but its silhouette
    // averages smooth along a camera ray; this band preserves multi-scale
    // cauliflower shoulders after projection without adding solid puffs.
    float silhouetteLow =
        fbm(glm::vec3(p.x * 3.6f + seedP.x * 0.19f,
                      p.y * 3.1f + seedP.y * 0.13f,
                      41.3f + seedP.z * 0.09f),
            4);
    float silhouetteMid =
        fbm(glm::vec3(p.x * 8.2f + seedP.x * 0.23f,
                      p.y * 6.5f + seedP.y * 0.11f,
                      71.7f + seedP.z * 0.07f),
            4);
    float silhouette = 0.65f * silhouetteLow + 0.35f * silhouetteMid;
    float coherentStrength =
        0.13f + 0.38f * height + 0.32f * (1.0f - mass);
    float silhouetteGate = glm::smoothstep(0.35f, 0.67f, silhouette);
    erosion += coherentStrength * (1.0f - silhouetteGate);
    float field =
        fmaxf(mass + 0.10f * (weather - 0.5f) - erosion, 0.0f);

    float density = glm::smoothstep(0.05f, 0.48f, field);
    float internal =
        0.48f + 0.52f *
                    glm::smoothstep(0.22f, 0.78f,
                                    0.65f * billow + 0.35f * detail);
    return glm::clamp(density * internal, 0.0f, 1.0f);
}

// Rising-plume profile: a continuous advected column around a wandering axis.
// Its radius broadens gradually, density dissipates with height, and
// anisotropic domain-warped noise tears the edge into sheets and wisps. This
// replaces the former max-union of sixteen swelling spheres.
__host__ __device__ inline float plumeDensity(const glm::vec3 &pl,
                                              float noiseScale, int octaves,
                                              int seed) {
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    glm::vec3 seedP = noiseSeedOffset(seed);

    float axisNoiseX =
        fbm(glm::vec3(h * 2.4f, 3.7f, 9.1f) + seedP * 0.21f, 3) - 0.5f;
    float axisNoiseZ =
        fbm(glm::vec3(h * 2.2f, 17.3f, 5.9f) + seedP * 0.21f, 3) - 0.5f;
    glm::vec2 center(
        (0.025f + 0.24f * h) * sinf(8.5f * h + 5.0f * axisNoiseX) +
            0.15f * h * axisNoiseX,
        (0.020f + 0.20f * h) * cosf(7.3f * h + 5.0f * axisNoiseZ) +
            0.13f * h * axisNoiseZ);
    glm::vec2 dp(pl.x - center.x, pl.z - center.y);

    // No giant terminal sphere: broadening is gradual, with a modest
    // turbulent shoulder before the plume dissipates.
    float shoulder = expf(-90.0f * (h - 0.72f) * (h - 0.72f));
    float radius = 0.060f + 0.170f * powf(h, 0.72f) + 0.070f * shoulder;
    float radial = glm::length(dp) / fmaxf(radius, 1e-4f);
    float envelope = 1.0f - glm::smoothstep(0.30f, 1.14f, radial);
    if (envelope <= 0.0f) {
        return 0.0f;
    }

    // Stretch noise along the rise axis to produce coherent wisps rather than
    // isotropic bubbles. `octaves` now genuinely controls plume detail.
    glm::vec3 q(dp.x * noiseScale * 1.35f, h * noiseScale * 0.48f,
                dp.y * noiseScale * 1.35f);
    q += seedP;
    glm::vec3 warp(
        fbm(q * 0.48f + glm::vec3(7.0f), 3),
        fbm(q * 0.48f + glm::vec3(23.0f), 3),
        fbm(q * 0.48f + glm::vec3(41.0f), 3));
    float detail =
        fbm(q + 1.25f * (warp - glm::vec3(0.5f)), octaves);
    float ribbon =
        0.5f + 0.5f * sinf(11.0f * h + 3.5f * atan2f(dp.y, dp.x) +
                           2.0f * (detail - 0.5f));
    float breakup =
        glm::smoothstep(0.27f + 0.18f * h, 0.72f, detail + 0.12f * ribbon);
    float d = envelope * (0.36f + 0.64f * breakup);

    // Source fade, upper dissipation, and mild stratified gaps.
    d *= glm::smoothstep(0.0f, 0.055f, h);
    d *= 1.0f - glm::smoothstep(0.82f, 1.0f, h);
    d *= 0.76f + 0.24f * sinf(34.0f * h + 4.0f * detail);
    return glm::clamp(d, 0.0f, 1.0f);
}

// Tapered, forked flame tongues. Elongated envelopes provide the macro shape;
// animated-looking turbulence comes from height-stretched domain warping.
__host__ __device__ inline float flameDensity(const glm::vec3 &pl,
                                              float noiseScale, int octaves,
                                              int seed) {
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    glm::vec3 seedP = noiseSeedOffset(seed);
    float wander = fbm(glm::vec3(h * 3.1f, 7.3f, 13.7f) + seedP * 0.19f, 3) -
                   0.5f;
    glm::vec2 center(0.09f * h * sinf(10.0f * h + 4.0f * wander),
                     0.055f * h * cosf(8.0f * h - 3.0f * wander));
    glm::vec2 dp(pl.x - center.x, pl.z - center.y);

    float radius = 0.245f * powf(fmaxf(1.0f - h, 0.0f), 0.68f) + 0.018f;
    float mainTongue =
        1.0f - glm::smoothstep(0.38f, 1.05f, glm::length(dp) / radius);

    // Two narrow upper tongues peel away from the core. They are stretched in
    // height, so even the max composition cannot read as spherical puffs.
    float forkGate = glm::smoothstep(0.38f, 0.62f, h);
    float forkRadius = 0.040f + 0.060f * (1.0f - h);
    glm::vec2 forkA(center.x + 0.12f * forkGate * (h - 0.35f),
                    center.y - 0.035f * forkGate);
    glm::vec2 forkB(center.x - 0.10f * forkGate * (h - 0.35f),
                    center.y + 0.045f * forkGate);
    float tongueA =
        (1.0f - glm::smoothstep(0.35f, 1.0f,
                                glm::length(glm::vec2(pl.x, pl.z) - forkA) /
                                    forkRadius)) *
        (1.0f - glm::smoothstep(0.88f, 1.0f, h));
    float tongueB =
        (1.0f - glm::smoothstep(0.35f, 1.0f,
                                glm::length(glm::vec2(pl.x, pl.z) - forkB) /
                                    forkRadius)) *
        (1.0f - glm::smoothstep(0.74f, 0.93f, h));
    float envelope =
        fmaxf(mainTongue, forkGate * fmaxf(tongueA, tongueB));
    if (envelope <= 0.0f) {
        return 0.0f;
    }

    glm::vec3 q(pl.x * noiseScale * 1.5f, h * noiseScale * 0.52f,
                pl.z * noiseScale * 1.5f);
    q += seedP;
    glm::vec3 warp(
        fbm(q * 0.55f + glm::vec3(5.1f), 3),
        fbm(q * 0.55f + glm::vec3(19.7f), 3),
        fbm(q * 0.55f + glm::vec3(43.3f), 3));
    float turbulence =
        fbm(q + 1.35f * (warp - glm::vec3(0.5f)), octaves);
    float lick = 0.5f + 0.5f * sinf(29.0f * h + 8.0f * turbulence);
    float d = envelope *
              glm::smoothstep(0.20f + 0.20f * h, 0.78f,
                              turbulence + 0.20f * lick + 0.18f * envelope);
    d *= glm::smoothstep(0.0f, 0.045f, h);
    d *= 1.0f - glm::smoothstep(0.92f, 1.0f, h);
    return glm::clamp(d, 0.0f, 1.0f);
}

// A candle flame is a mostly laminar teardrop, not a miniature turbulent
// campfire. These helpers are shared by density and emission so the authored
// dark wick zone and temperature gradient stay registered to the silhouette.
__host__ __device__ inline glm::vec2
candleFlameCenter(float h, const glm::vec3 &seedP) {
    float phase = 0.13f * seedP.x + 0.07f * seedP.z;
    return glm::vec2(0.018f * sinf(4.2f * h + phase) +
                         0.032f * h * h * sinf(phase),
                     0.010f * cosf(3.4f * h - phase));
}

__host__ __device__ inline float candleFlameRadius(float h) {
    float rise = glm::smoothstep(0.0f, 0.17f, h);
    float taper = powf(fmaxf(1.0f - h, 0.0f), 0.58f);
    return 0.018f + 0.215f * rise * taper;
}

__host__ __device__ inline float candleFlameDensity(const glm::vec3 &pl,
                                                    float noiseScale,
                                                    int octaves, int seed) {
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    glm::vec3 seedP = noiseSeedOffset(seed);
    glm::vec2 dp =
        glm::vec2(pl.x, pl.z) - candleFlameCenter(h, seedP);
    float radial = glm::length(dp) / fmaxf(candleFlameRadius(h), 1e-4f);

    float envelope = 1.0f - glm::smoothstep(0.58f, 1.04f, radial);
    envelope *= glm::smoothstep(0.0f, 0.035f, h);
    envelope *= 1.0f - glm::smoothstep(0.93f, 1.0f, h);
    if (envelope <= 0.0f) {
        return 0.0f;
    }

    // Only subtle low-frequency variation: the body must remain continuous
    // and smooth while the tip leans slightly. A dark lower-center void makes
    // the wick-rooted reaction zone visible instead of producing a glowing
    // disk at the source.
    glm::vec3 q(dp.x * noiseScale, h * noiseScale * 0.30f,
                dp.y * noiseScale);
    float detail =
        fbm(q + seedP, min(octaves, 3));
    float darkCore =
        (1.0f - glm::smoothstep(0.12f, 0.62f, radial)) *
        (1.0f - glm::smoothstep(0.18f, 0.43f, h));
    float d = envelope * (0.86f + 0.14f * detail);
    d *= 1.0f - 0.72f * darkCore;
    return glm::clamp(d, 0.0f, 1.0f);
}

// Connected combustion-front field. Real wood and vegetation fires are thin,
// folded reaction sheets surrounding fuel vapour, not a row of solid tubes.
// The broad density and the much narrower hot-ribbon field are evaluated
// together so extinction, emission, and temperature stay spatially aligned.
//
// `density` and `hotRibbon` are both normalized to [0, 1]. Hot ribbons drive
// the hotter source term and add a small extinction support before the final
// density clamp, so they remain covered by the same unit-density majorant.
__host__ __device__ inline float fireFrontUnion(float a, float b) {
    a = glm::clamp(a, 0.0f, 1.0f);
    b = glm::clamp(b, 0.0f, 1.0f);
    return a + b - a * b;
}

// Accumulate one warped signed-distance flame sheet. `phi` is an approximate
// local distance to a two-dimensional lamella: its broad band contributes the
// translucent orange body, while its center/edge and perforation contours
// contribute much narrower hot ribbons.
__host__ __device__ inline void
fireFrontLamella(float x, float z, float h, float theta, float offset,
                 float sheetPhase, float warpX, float warpZ, float weather,
                 float detail, float holeNoise, float topMask,
                 float contourWeight,
                 float &body, float &ribbon) {
    float ct = cosf(theta);
    float st = sinf(theta);
    float u = ct * x + st * z;
    float v = -st * x + ct * z;
    float projectedWarp = ct * warpZ - st * warpX;
    float surface =
        offset +
        (0.050f + 0.140f * h) *
            (1.15f * projectedWarp +
             0.36f * sinf(5.1f * u + 6.7f * h + sheetPhase)) +
        0.034f * h * h *
            sinf(11.3f * h - 3.1f * u + 0.73f * sheetPhase);
    // Smaller turbulent folds perturb the sheet itself instead of merely
    // modulating its opacity. Their amplitude grows away from the anchored
    // fuel bed, breaking the long parallel edges that read as swept quads.
    surface +=
        (0.008f + 0.034f * h) * (2.0f * detail - 1.0f) +
        0.010f * h *
            sinf(18.7f * h + 7.3f * u + 0.41f * sheetPhase);
    float phi = v - surface;
    float absPhi = fabsf(phi);

    // Vary the local sheet width on a finer scale than the gross fold. Real
    // reaction sheets repeatedly pinch, flare, and fork instead of retaining
    // parallel sides.
    float sheetWidth =
        glm::clamp(0.050f +
                       0.014f *
                           sinf(15.1f * h - 6.2f * u +
                                0.57f * sheetPhase) +
                       0.018f * (detail - 0.5f) +
                       0.010f * (holeNoise - 0.5f),
                   0.028f, 0.072f);
    float broad =
        1.0f -
        glm::smoothstep(0.34f * sheetWidth, sheetWidth + 0.024f,
                        absPhi);
    float lowJoin = 1.0f - glm::smoothstep(0.12f, 0.28f, h);
    float crown =
        glm::smoothstep(0.30f + 0.22f * h, 0.69f,
                        0.46f * weather + 0.34f * detail +
                            0.20f * holeNoise);
    float active =
        topMask * (0.22f + 0.78f * fmaxf(lowJoin, crown));

    float holeSignal = 0.62f * holeNoise + 0.38f * detail;
    float perforation =
        glm::smoothstep(0.40f + 0.08f * h, 0.63f, holeSignal);
    perforation = fmaxf(lowJoin, perforation);
    float sheetBody =
        0.92f * broad * active * (0.10f + 0.90f * perforation) *
        (0.62f + 0.38f * detail);

    float outerRim =
        1.0f -
        glm::smoothstep(0.0035f, 0.012f,
                        fabsf(absPhi - 0.82f * sheetWidth));
    float coreLine =
        1.0f - glm::smoothstep(0.0035f, 0.013f, absPhi);
    float holeContour =
        1.0f -
        glm::smoothstep(0.014f, 0.050f,
                        fabsf(holeSignal - (0.515f + 0.055f * h)));
    float streamCenterA =
        0.15f * sinf(4.8f * h + 0.37f * sheetPhase) +
        0.055f *
            sinf(12.6f * h - 1.9f * u + 0.63f * sheetPhase) +
        0.12f * projectedWarp + 0.040f * (holeNoise - 0.5f) +
        0.020f *
            sinf(9.4f * h + 3.7f * u + 0.83f * sheetPhase);
    float streamCenterB =
        -0.24f + 0.22f * h +
        0.10f * sinf(5.6f * h + 0.71f * sheetPhase) -
        0.06f * projectedWarp +
        0.017f * (detail - holeNoise);
    float streamA =
        1.0f -
        glm::smoothstep(0.008f, 0.038f,
                        fabsf(u - streamCenterA));
    float streamB =
        1.0f -
        glm::smoothstep(0.009f, 0.042f,
                        fabsf(u - streamCenterB));
    float streamGate = fmaxf(streamA, 0.68f * streamB);
    float sheetRibbon =
        active *
        fmaxf(0.62f * outerRim * perforation * streamGate,
              fmaxf(0.74f * coreLine * perforation * streamGate,
                    contourWeight * broad * holeContour));

    body = fireFrontUnion(body, sheetBody);
    ribbon = fireFrontUnion(ribbon, sheetRibbon);
}

__host__ __device__ inline float
fireFrontWindow(float x, float h, float centerX, float centerH,
                float radiusX, float radiusH, float noise, float phase) {
    float dx = (x - centerX) / radiusX;
    float dy = (h - centerH) / radiusH;
    dx += 0.24f * dy * dy * sinf(phase + 2.1f * dy);
    float distance =
        sqrtf(dx * dx + dy * dy) +
        0.20f * noise +
        0.10f * sinf(4.3f * dx - 3.7f * dy + phase) +
        0.035f * sinf(9.1f * dx + 6.3f * dy - 0.7f * phase);
    return 1.0f - glm::smoothstep(0.72f, 1.04f, distance);
}

// A narrow, depth-warped reaction stream carried inside the broader orange
// flame body. These curves are emission structure, not independent flame
// tongues: the body gate below prevents them from floating away from fuel.
__host__ __device__ inline float
fireFrontHotStream(float x, float z, float h, float rootX, float phase,
                   float top, float width, float zOffset, float warpX,
                   float warpZ, float carrierNoise) {
    float centerX =
        rootX * (1.0f - 0.22f * h) +
        (0.022f + 0.105f * h) *
            sinf(4.2f * h + phase + 0.55f * warpX) +
        0.026f * sinf(13.7f * h - 1.3f * phase) +
        0.024f * h * (2.0f * carrierNoise - 1.0f);
    float widthPulse =
        0.45f +
        0.90f *
            (0.5f +
             0.5f * sinf(10.3f * h + 1.7f * phase));
    float feather =
        fmaxf(0.004f, 1.05f * width);
    float xBand =
        1.0f -
        glm::smoothstep(width * widthPulse,
                        width * widthPulse + feather,
                        fabsf(x - centerX));
    float centerZ =
        zOffset +
        (0.030f + 0.105f * h) *
            (0.72f * warpZ +
             0.28f * sinf(5.3f * h - 0.9f * phase));
    float zBand =
        1.0f -
        glm::smoothstep(0.035f, 0.135f, fabsf(z - centerZ));
    float heightGate =
        glm::smoothstep(0.025f, 0.095f, h) *
        (1.0f - glm::smoothstep(top - 0.11f, top + 0.025f, h));
    float brokenFlow =
        0.34f +
        0.66f *
            glm::smoothstep(
                0.18f, 0.72f,
                0.62f * carrierNoise +
                    0.38f *
                        (0.5f +
                         0.5f *
                             sinf(17.1f * h + 2.1f * phase)));
    return xBand * zBand * heightGate * brokenFlow;
}

__host__ __device__ inline void
fireFrontFields(const glm::vec3 &pl, float noiseScale, int octaves, int seed,
                float aspectXY, float &density, float &hotRibbon) {
    density = 0.0f;
    hotRibbon = 0.0f;
    aspectXY = glm::clamp(aspectXY, 0.65f, 4.0f);
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    if (h <= 0.0f || h >= 1.0f) {
        return;
    }

    glm::vec3 seedP = noiseSeedOffset(seed);
    float phase = 0.11f * seedP.x + 0.17f * seedP.z;

    // Low-frequency 3-D domain warp bends the complete reaction sheet. The
    // displacement grows with height, leaving the fuel contact anchored while
    // letting the upper sheet roll and lean.
    glm::vec3 macroQ(pl.x * aspectXY * noiseScale * 0.46f,
                     h * noiseScale * 0.24f,
                     pl.z * noiseScale * 0.42f);
    macroQ += seedP;
    float warpX = fbm(macroQ * 0.52f + glm::vec3(5.3f), 3) - 0.5f;
    float warpZ = fbm(macroQ * 0.52f + glm::vec3(23.7f), 3) - 0.5f;
    glm::vec3 p(
        pl.x + (0.025f + 0.13f * h * h) * warpX,
        pl.y,
        pl.z + (0.020f + 0.11f * h * h) * warpZ);

    // A coherent X/Z weather field controls both irregular root height and
    // the large 2--4x variation in peak height. It is intentionally coherent
    // through Y; fully 3-D noise alone averages into a uniform picket line in
    // projection.
    float weatherFrequency =
        glm::clamp(0.55f * noiseScale, 3.2f, 6.4f);
    float weather =
        fbm(glm::vec3(p.x * aspectXY * weatherFrequency +
                          seedP.x * 0.17f,
                      p.z * (0.88f * weatherFrequency) +
                          seedP.z * 0.13f,
                      17.1f + seedP.y * 0.11f),
            4);
    float fineFuel =
        valueNoise(glm::vec3(p.x * aspectXY * 8.7f, p.z * 7.9f,
                             43.7f + 0.19f * seedP.y));
    float heightSignal =
        glm::smoothstep(0.31f, 0.68f,
                        0.82f * weather + 0.18f * fineFuel);
    float topHeight =
        0.13f + 0.84f * powf(heightSignal, 1.35f);
    float topMask =
        1.0f - glm::smoothstep(topHeight - 0.085f,
                               topHeight + 0.035f, h);
    float sideLimit =
        0.47f - 0.11f * h +
        0.035f * (0.62f * weather + 0.38f * fineFuel - 0.5f) +
        0.020f * sinf(8.0f * h + phase);
    float lateralMask =
        1.0f - glm::smoothstep(sideLimit - 0.075f,
                               sideLimit + 0.025f, fabsf(p.x));
    topMask *= lateralMask;
    // Restrict fine erosion to the crown and exposed sides. The central lower
    // mass remains connected while tips acquire forks, notches, and torn
    // edges instead of a smooth curtain silhouette.
    float edgeDetail =
        valueNoise(glm::vec3(p.x * aspectXY * 15.7f +
                                 seedP.z * 0.05f,
                             h * 13.9f + seedP.x * 0.06f,
                             91.3f + seedP.y * 0.04f));
    float crownBand =
        glm::smoothstep(topHeight - 0.24f, topHeight + 0.015f, h);
    float sideBand =
        glm::smoothstep(sideLimit - 0.10f, sideLimit + 0.015f,
                        fabsf(p.x));
    float exposedEdge = fmaxf(crownBand, 0.72f * sideBand);
    float edgeSurvival =
        glm::smoothstep(0.24f, 0.68f,
                        0.68f * edgeDetail + 0.32f * fineFuel);
    topMask *= 1.0f - 0.78f * exposedEdge *
                           (1.0f - edgeSurvival);

    // A continuous fuel-bed reaction region fixes the floating roots, but its
    // noisy top and partial holes prevent a ruler-straight emissive strip.
    float rootHeight =
        0.055f + 0.105f * weather + 0.035f * (fineFuel - 0.5f);
    float rootLayer =
        glm::smoothstep(0.0f, 0.025f, h) *
        (1.0f - glm::smoothstep(rootHeight - 0.025f,
                                rootHeight + 0.045f, h));
    float fuelCoverage =
        glm::smoothstep(0.20f, 0.57f,
                        0.66f * weather + 0.34f * fineFuel);
    float rootCenterZ =
        0.085f *
            (2.0f *
                 valueNoise(glm::vec3(p.x * aspectXY * 2.3f,
                                      11.7f + seedP.x * 0.09f,
                                      4.7f + seedP.z * 0.07f)) -
             1.0f) +
        0.025f * sinf(7.0f * aspectXY * p.x + phase);
    float rootHalfDepth = 0.27f + 0.08f * weather;
    float rootDepth =
        1.0f -
        glm::smoothstep(rootHalfDepth, rootHalfDepth + 0.075f,
                        fabsf(p.z - rootCenterZ));
    float connectedRoot =
        rootLayer * rootDepth * (0.20f + 0.80f * fuelCoverage);
    float rootIslands =
        glm::smoothstep(
            0.24f, 0.72f,
            0.50f +
                0.34f *
                    sinf((12.7f + 0.62f * noiseScale) *
                             aspectXY * p.x +
                         3.1f * weather + phase) +
                0.16f * (fineFuel - 0.5f));
    float lowRootBand =
        1.0f - glm::smoothstep(0.11f, 0.23f, h);
    float rootSlots =
        glm::smoothstep(
            0.72f, 0.94f,
            fabsf(sinf((9.0f + 0.35f * noiseScale) *
                           aspectXY * p.x +
                       2.2f * weather + 0.7f * phase))) *
        lowRootBand;
    // A dim residual bridge keeps the bed topologically connected, while the
    // brighter islands expose distinct attachment points at the fuel.
    connectedRoot *=
        (0.10f + 0.90f * rootIslands) *
        (1.0f - 0.82f * rootSlots);

    // Seven seed-varying signed-distance lamellae span the connected fuel bed
    // at different depths. They remain individual folded surfaces rather than
    // periodic sine level sets, avoiding the crosshatched lattice that the
    // first sheet prototype produced.
    glm::vec3 detailQ(p.x * aspectXY * noiseScale * 1.12f,
                      h * noiseScale * 0.40f,
                      p.z * noiseScale * 1.04f);
    detailQ += seedP;
    float detail =
        fbm(detailQ +
                glm::vec3(1.25f * warpX, 0.0f, 1.25f * warpZ),
            octaves);
    float tear =
        valueNoise(detailQ * 1.83f + glm::vec3(31.1f, 11.7f, 53.9f));
    float hole1 =
        valueNoise(detailQ * 1.47f + glm::vec3(7.3f, 47.9f, 19.1f));
    float hole2 =
        valueNoise(detailQ * 1.61f + glm::vec3(61.7f, 3.1f, 37.3f));
    float hole3 =
        valueNoise(detailQ * 1.73f + glm::vec3(13.9f, 71.3f, 5.7f));
    float hole4 =
        valueNoise(detailQ * 1.39f + glm::vec3(43.1f, 29.7f, 79.3f));
    float hole5 =
        valueNoise(detailQ * 1.57f + glm::vec3(83.9f, 17.3f, 23.5f));
    float hole6 =
        valueNoise(detailQ * 1.69f + glm::vec3(3.7f, 59.1f, 67.7f));

    float sheetBody = 0.0f;
    float sheetRibbon = 0.0f;
    // Lamella-local highlight gates work at hearth scale, but their projected
    // width grows with a very wide container and can become a detached
    // horizontal lozenge. Fade those secondary highlights out on a wide
    // front; the separately evaluated, transform-scaled fuel-rooted streams
    // below remain active.
    float lamellaRibbonWeight =
        1.0f - glm::smoothstep(1.60f, 2.00f, aspectXY);
    float contourWeight = 0.10f * lamellaRibbonWeight;
    fireFrontLamella(p.x, p.z, h, 0.42f, -0.41f, phase - 1.2f,
                     warpX, warpZ, weather, detail, hole5, topMask,
                     contourWeight,
                     sheetBody, sheetRibbon);
    fireFrontLamella(p.x, p.z, h, 0.06f, -0.30f, phase + 0.4f,
                     warpX, warpZ, weather, detail, tear, topMask,
                     contourWeight,
                     sheetBody, sheetRibbon);
    fireFrontLamella(p.x, p.z, h, -0.24f, -0.15f, phase + 1.8f,
                     warpX, warpZ, weather, detail, hole1, topMask,
                     contourWeight,
                     sheetBody, sheetRibbon);
    fireFrontLamella(p.x, p.z, h, 0.18f, 0.00f, phase + 3.2f,
                     warpX, warpZ, weather, detail, hole2, topMask,
                     contourWeight,
                     sheetBody, sheetRibbon);
    fireFrontLamella(p.x, p.z, h, -0.36f, 0.16f, phase + 4.9f,
                     warpX, warpZ, weather, detail, hole3, topMask,
                     contourWeight,
                     sheetBody, sheetRibbon);
    fireFrontLamella(p.x, p.z, h, 0.31f, 0.31f, phase + 6.7f,
                     warpX, warpZ, weather, detail, hole4, topMask,
                     contourWeight,
                     sheetBody, sheetRibbon);
    fireFrontLamella(p.x, p.z, h, -0.48f, 0.41f, phase + 8.4f,
                     warpX, warpZ, weather, detail, hole6, topMask,
                     contourWeight,
                     sheetBody, sheetRibbon);
    sheetRibbon *= lamellaRibbonWeight;

    // A broader turbulent envelope joins nearby folded sheets. It is still a
    // true three-dimensional field with a warped depth band and noisy
    // coverage, but it supplies the translucent orange mass in which the
    // much narrower reaction ribbons live.
    float bulkNoise =
        fbm(glm::vec3(p.x * aspectXY * 5.8f +
                          seedP.x * 0.09f,
                      h * 4.7f + seedP.y * 0.07f,
                      37.3f + seedP.z * 0.11f),
            4);
    float bulkFine =
        valueNoise(glm::vec3(p.x * aspectXY * 12.1f +
                                 seedP.z * 0.05f,
                             h * 10.7f + seedP.x * 0.06f,
                             73.1f + seedP.y * 0.08f));
    float bulkSignal = 0.68f * bulkNoise + 0.32f * bulkFine;
    float bulkCoverage =
        glm::smoothstep(0.29f + 0.10f * h,
                        0.57f + 0.11f * h, bulkSignal);
    float bulkCenterZ =
        0.075f * warpZ +
        0.026f * sinf(6.1f * p.x + 4.3f * h + phase);
    float bulkDepth =
        1.0f -
        glm::smoothstep(0.22f + 0.045f * weather,
                        0.40f + 0.055f * weather,
                        fabsf(p.z - bulkCenterZ));
    float lowerBridge =
        (1.0f - glm::smoothstep(0.13f, 0.29f, h)) *
        (0.05f + 0.55f * rootIslands) *
        (1.0f - 0.78f * rootSlots);
    float bulkBody =
        0.72f * topMask * bulkDepth *
        fmaxf(bulkCoverage, lowerBridge) *
        (0.66f + 0.34f * detail);
    sheetBody = fireFrontUnion(sheetBody, bulkBody);

    // Large, irregular windows cross the lamella stack in image-space. They
    // leave bridges around their rims, producing the hooked/crescent voids
    // visible inside a dense real flame rather than only gaps between sheets.
    float holeWarp =
        2.0f *
            valueNoise(glm::vec3(p.x * aspectXY * 4.6f +
                                     seedP.x * 0.07f,
                                 h * 5.8f + seedP.y * 0.05f,
                                 29.3f + seedP.z * 0.09f)) -
        1.0f;
    float featureScale =
        glm::clamp(8.8f / fmaxf(noiseScale, 1.0f), 0.56f, 1.05f);
    float metricX = 1.0f / aspectXY;
    float hx = p.x + 0.052f * holeWarp +
               0.020f * h * warpX;
    float hh = h + 0.030f * (detail - 0.5f) +
               0.012f * warpZ;
    float holeA =
        fireFrontWindow(hx, hh, -0.235f, 0.41f,
                        0.046f * featureScale * metricX,
                        0.115f * featureScale,
                        holeWarp, phase + 0.3f);
    float holeB =
        fireFrontWindow(hx, hh, -0.020f, 0.61f,
                        0.090f * featureScale * metricX,
                        0.070f * featureScale,
                        holeWarp, phase + 2.4f);
    float holeBInner =
        fireFrontWindow(hx, hh, 0.010f, 0.625f,
                        0.066f * featureScale * metricX,
                        0.052f * featureScale,
                        holeWarp, phase + 2.9f);
    holeB *= 1.0f - 0.82f * holeBInner;
    float holeC =
        fireFrontWindow(hx, hh, 0.205f, 0.34f,
                        0.074f * featureScale * metricX,
                        0.098f * featureScale,
                        holeWarp, phase + 3.7f);
    holeC *= 0.40f;
    float holeD =
        fireFrontWindow(hx, hh, 0.095f, 0.78f,
                        0.040f * featureScale * metricX,
                        0.068f * featureScale,
                        holeWarp, phase + 5.4f);
    float windowCut =
        fireFrontUnion(fireFrontUnion(holeA, holeB),
                       fireFrontUnion(holeC, holeD));
    sheetBody *= 1.0f - 0.88f * windowCut;
    sheetRibbon *= 1.0f - 0.96f * windowCut;

    // Several independently bending inner streams restore the real
    // yellow-white reaction hierarchy without turning the entire orange
    // envelope up. Their roots share the same connected fuel bed and their
    // visibility is clipped to the windowed body.
    float streamWidth =
        0.0105f * featureScale * metricX;
    float innerStreams = 0.0f;
    innerStreams = fireFrontUnion(
        innerStreams,
        fireFrontHotStream(p.x, p.z, h, -0.34f, phase + 0.2f, 0.64f,
                           streamWidth, -0.13f, warpX, warpZ, hole1));
    innerStreams = fireFrontUnion(
        innerStreams,
        fireFrontHotStream(p.x, p.z, h, -0.17f, phase + 1.5f, 0.88f,
                           0.90f * streamWidth, 0.08f, warpX, warpZ,
                           hole3));
    innerStreams = fireFrontUnion(
        innerStreams,
        fireFrontHotStream(p.x, p.z, h, 0.00f, phase + 2.8f, 0.96f,
                           1.12f * streamWidth, -0.03f, warpX, warpZ,
                           tear));
    innerStreams = fireFrontUnion(
        innerStreams,
        fireFrontHotStream(p.x, p.z, h, 0.18f, phase + 4.4f, 0.76f,
                           0.86f * streamWidth, 0.13f, warpX, warpZ,
                           hole4));
    innerStreams = fireFrontUnion(
        innerStreams,
        fireFrontHotStream(
            p.x, p.z, h, 0.34f, phase + 5.7f,
            0.45f + 0.13f * lamellaRibbonWeight,
            streamWidth, -0.10f, warpX, warpZ, hole6));
    innerStreams *=
        glm::smoothstep(0.045f, 0.26f, sheetBody) *
        (1.0f - windowCut);
    sheetRibbon = fireFrontUnion(sheetRibbon, innerStreams);
    // Open the dominant central arch on one side. The noisy attenuation is
    // deliberately asymmetric, so no two surviving streams form a closed,
    // mirrored handle.
    float loopBreak =
        (1.0f -
         glm::smoothstep(0.035f * metricX, 0.125f * metricX,
                         fabsf(p.x - 0.055f))) *
        glm::smoothstep(0.50f, 0.60f, h) *
        (1.0f - glm::smoothstep(0.73f, 0.82f, h)) *
        (0.58f + 0.42f * edgeDetail);
    sheetRibbon *= 1.0f - 0.86f * loopBreak;

    // A separate ragged reaction line lies just above the fuel. This produces
    // bright contacts around logs/brush rather than a uniformly bright base.
    float rootLineHeight =
        0.060f + 0.060f * weather + 0.018f * (fineFuel - 0.5f);
    float rootRibbon =
        1.0f -
        glm::smoothstep(0.012f, 0.050f,
                        fabsf(h - rootLineHeight));
    float rootStream =
        1.0f -
        glm::smoothstep(0.06f, 0.28f,
                        fabsf(sinf(11.0f * p.x +
                                   4.0f * weather + phase)));
    rootRibbon *= fuelCoverage * rootLayer * rootDepth *
                  (0.18f + 0.82f * rootStream);

    hotRibbon = fireFrontUnion(sheetRibbon, 0.22f * rootRibbon);
    hotRibbon = glm::clamp(hotRibbon, 0.0f, 1.0f);
    density = fireFrontUnion(
        connectedRoot,
        fireFrontUnion(0.88f * sheetBody, 0.24f * hotRibbon));
    density *= 1.0f - glm::smoothstep(0.96f, 1.0f, h);
    density = glm::clamp(density, 0.0f, 1.0f);
}

__host__ __device__ inline float
fireFrontDensity(const glm::vec3 &pl, float noiseScale, int octaves,
                 int seed, float aspectXY) {
    float density, hotRibbon;
    fireFrontFields(pl, noiseScale, octaves, seed, aspectXY, density,
                    hotRibbon);
    return density;
}

// One widening convective smoke source. Multiple sources are smooth-summed
// by smokeFrontDensity so their dark stems merge into an irregular canopy.
__host__ __device__ inline float
smokeFrontUpdraft(const glm::vec3 &pl, float h, float baseX, float baseZ,
                  float phase, float weight) {
    glm::vec2 center(
        baseX + (0.025f + 0.12f * h) *
                    sinf(5.4f * h + phase),
        baseZ + (0.020f + 0.095f * h) *
                    cosf(4.8f * h - phase));
    float shoulder = expf(-55.0f * (h - 0.70f) * (h - 0.70f));
    float radius =
        0.060f + 0.155f * powf(h, 0.72f) + 0.055f * shoulder;
    float radial =
        glm::length(glm::vec2(pl.x, pl.z) - center) /
        fmaxf(radius, 1e-4f);
    return weight * (1.0f - glm::smoothstep(0.34f, 1.08f, radial));
}

// Merged multi-source smoke for a broad combustion front. Lower stems remain
// distinguishable; widening updrafts and a noisy canopy coalesce aloft.
__host__ __device__ inline float
smokeFrontDensity(const glm::vec3 &pl, float noiseScale, int octaves,
                  int seed) {
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    glm::vec3 seedP = noiseSeedOffset(seed);
    float phase = 0.09f * seedP.x + 0.14f * seedP.z;

    float sum = 0.0f;
    sum += smokeFrontUpdraft(pl, h, -0.38f, 0.05f, phase + 0.2f, 0.80f);
    sum += smokeFrontUpdraft(pl, h, -0.19f, -0.07f, phase + 1.8f, 1.05f);
    sum += smokeFrontUpdraft(pl, h, 0.00f, 0.08f, phase + 3.5f, 1.15f);
    sum += smokeFrontUpdraft(pl, h, 0.20f, -0.05f, phase + 5.2f, 1.00f);
    sum += smokeFrontUpdraft(pl, h, 0.39f, 0.06f, phase + 6.9f, 0.85f);

    float canopy =
        expf(-13.0f * (h - 0.72f) * (h - 0.72f)) *
        expf(-2.4f * (pl.x * pl.x / (0.48f * 0.48f) +
                     pl.z * pl.z / (0.39f * 0.39f)));
    float mass = 1.0f - expf(-(sum + 0.45f * canopy));
    if (mass <= 0.0f) {
        return 0.0f;
    }

    glm::vec3 q(pl.x * noiseScale * 1.10f, h * noiseScale * 0.34f,
                pl.z * noiseScale);
    q += seedP;
    glm::vec3 warp(
        fbm(q * 0.46f + glm::vec3(7.0f), 3),
        fbm(q * 0.46f + glm::vec3(29.0f), 3),
        fbm(q * 0.46f + glm::vec3(53.0f), 3));
    float detail =
        fbm(q + 1.15f * (warp - glm::vec3(0.5f)), octaves);
    float ribbon =
        0.5f + 0.5f *
                   sinf(15.0f * h + 10.0f * pl.x +
                        4.0f * (detail - 0.5f));
    float breakup =
        glm::smoothstep(0.25f + 0.12f * h, 0.73f,
                        detail + 0.15f * ribbon + 0.18f * mass);
    float d = mass * (0.03f + 0.97f * breakup);
    d *= glm::smoothstep(0.0f, 0.045f, h);
    d *= 1.0f - glm::smoothstep(0.87f, 1.0f, h);
    return glm::clamp(d, 0.0f, 1.0f);
}

// A soft atmospheric bank with denser low layers, broad low-frequency
// variation, and sparse lifted wisps. The enclosing cube fade in
// mediumDensity is widened for this profile so its boundary never reads as a
// glass box.
__host__ __device__ inline float fogDensity(const glm::vec3 &pl,
                                            float noiseScale, int octaves,
                                            int seed) {
    glm::vec3 seedP = noiseSeedOffset(seed);
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    glm::vec3 q(pl.x * noiseScale, h * noiseScale * 0.30f,
                pl.z * noiseScale);
    q += seedP;
    float broad = fbm(q * 0.42f, min(octaves, 4));
    float detail = fbm(q * 1.15f + glm::vec3(17.0f), octaves);
    // Ground fog should dissolve well before the container ceiling.  Leaving
    // appreciable density until h ~= 1 made even a faded AABB read as a
    // horizontal slab in back-lit scenes.
    float height = 1.0f - glm::smoothstep(0.46f, 0.76f, h);
    float lifted = expf(-45.0f * (h - 0.62f) * (h - 0.62f)) *
                   glm::smoothstep(0.52f, 0.78f, broad);
    float d = height * (0.48f + 0.42f * broad + 0.10f * detail) +
              0.20f * lifted;
    return glm::clamp(d, 0.0f, 1.0f);
}

// Soft local-space fade for shaped density and emission fields. Keeping this
// separate lets FireFront reuse one joint density/ribbon evaluation for its
// source term instead of recomputing every noise octave through mediumDensity.
__host__ __device__ inline float
mediumBoundaryFade(const Material &m, const Geom &g,
                   const glm::vec3 &pl) {
    if (g.type == SPHERE) {
        float r = glm::length(pl);
        return glm::clamp((0.5f - r) / 0.10f, 0.0f, 1.0f);
    }

    glm::vec3 a = glm::vec3(0.5f) - glm::abs(pl);
    float edge = fminf(a.x, fminf(a.y, a.z));
    float edgeWidth = m.mediumProfile == MEDIUM_PROFILE_FOG
                          ? 0.13f
                          : (m.mediumProfile == MEDIUM_PROFILE_SMOKE_FRONT
                                 ? 0.08f
                                 : ((m.mediumProfile ==
                                         MEDIUM_PROFILE_CLOUD ||
                                     m.mediumProfile ==
                                         MEDIUM_PROFILE_FIRE_FRONT)
                                        ? 0.10f
                                        : 0.04f));
    return glm::clamp(edge / edgeWidth, 0.0f, 1.0f);
}

// Normalized density in [0, 1] at a world point inside medium geom g.
// Homogeneous media are constant 1 (densityScale is folded into the sigma
// arrays by the caller). Heterogeneous media evaluate the material's density
// profile in the geom's local unit space, with a soft fade toward the
// boundary so the silhouette never shows the bounding cube/sphere.
__host__ __device__ inline float mediumDensity(const Material &m, const Geom &g,
                                               const glm::vec3 &pWorld) {
    if (!m.heterogeneous) {
        return 1.0f;
    }
    glm::vec3 pl =
        multiplyMV(g.transform.inverseTransform, glm::vec4(pWorld, 1.0f));

    float d;
    switch (m.mediumProfile) {
    case MEDIUM_PROFILE_CLOUD:
        d = cloudDensity(pl, m.noiseScale, m.noiseOctaves, m.noiseSeed);
        break;
    case MEDIUM_PROFILE_PLUME:
        d = plumeDensity(pl, m.noiseScale, m.noiseOctaves, m.noiseSeed);
        break;
    case MEDIUM_PROFILE_FLAME:
        d = flameDensity(pl, m.noiseScale, m.noiseOctaves, m.noiseSeed);
        break;
    case MEDIUM_PROFILE_CANDLE:
        d = candleFlameDensity(pl, m.noiseScale, m.noiseOctaves,
                               m.noiseSeed);
        break;
    case MEDIUM_PROFILE_FIRE_FRONT: {
        float scaleX =
            glm::length(glm::vec3(g.transform.transform[0]));
        float scaleY =
            glm::length(glm::vec3(g.transform.transform[1]));
        d = fireFrontDensity(pl, m.noiseScale, m.noiseOctaves,
                             m.noiseSeed,
                             scaleX / fmaxf(scaleY, 1e-5f));
        break;
    }
    case MEDIUM_PROFILE_SMOKE_FRONT:
        d = smokeFrontDensity(pl, m.noiseScale, m.noiseOctaves,
                              m.noiseSeed);
        break;
    case MEDIUM_PROFILE_FOG:
        d = fogDensity(pl, m.noiseScale, m.noiseOctaves, m.noiseSeed);
        break;
    case MEDIUM_PROFILE_FBM:
    default:
        // Thresholded fbm: carve empty pockets, rescale the rest to [0, 1].
        d = fbm(pl * m.noiseScale + noiseSeedOffset(m.noiseSeed),
                m.noiseOctaves);
        d = glm::clamp((d - 0.42f) / 0.33f, 0.0f, 1.0f);
        break;
    }

    return d * mediumBoundaryFade(m, g, pl);
}

// Spatial source strength and relative temperature for continuous medium
// emission. Fire is hottest at its dense lower core and cooler toward its
// thin upper/outer tongues. Other emissive profiles simply follow density.
__host__ __device__ inline void
mediumEmissionProperties(const Material &m, const Geom &g,
                         const glm::vec3 &pWorld, float &source,
                         float &relativeTemperature) {
    relativeTemperature = 1.0f;
    bool isFire = m.mediumProfile == MEDIUM_PROFILE_FLAME ||
                  m.mediumProfile == MEDIUM_PROFILE_CANDLE ||
                  m.mediumProfile == MEDIUM_PROFILE_FIRE_FRONT;

    // FireFront uses one joint evaluation so its narrow hot reaction ribbons
    // stay inside the broader extinction field. Emission may be more strongly
    // concentrated than extinction; this is the spatial source coefficient,
    // not a tracking probability, so it need not be bounded by the majorant.
    if (m.mediumProfile == MEDIUM_PROFILE_FIRE_FRONT) {
        glm::vec3 pl =
            multiplyMV(g.transform.inverseTransform,
                       glm::vec4(pWorld, 1.0f));
        float scaleX =
            glm::length(glm::vec3(g.transform.transform[0]));
        float scaleY =
            glm::length(glm::vec3(g.transform.transform[1]));
        float hotRibbon;
        fireFrontFields(pl, m.noiseScale, m.noiseOctaves, m.noiseSeed,
                        scaleX / fmaxf(scaleY, 1e-5f), source,
                        hotRibbon);
        source *= mediumBoundaryFade(m, g, pl);
        if (source <= 0.0f) {
            return;
        }

        float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
        float core = sqrtf(glm::clamp(source, 0.0f, 1.0f));
        relativeTemperature =
            glm::clamp(0.84f + 0.08f * core - 0.08f * h +
                           0.30f * hotRibbon,
                       0.62f, 1.15f);
        float baseAttenuation =
            0.54f + 0.46f * glm::smoothstep(0.075f, 0.25f, h);
        source *= baseAttenuation *
                  (0.36f + 0.24f * (1.0f - h) +
                   3.35f * hotRibbon);
        return;
    }

    source = mediumDensity(m, g, pWorld);
    if (!isFire || source <= 0.0f) {
        return;
    }

    glm::vec3 pl =
        multiplyMV(g.transform.inverseTransform, glm::vec4(pWorld, 1.0f));
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    float core = sqrtf(glm::clamp(source, 0.0f, 1.0f));
    if (m.mediumProfile == MEDIUM_PROFILE_CANDLE) {
        glm::vec3 seedP = noiseSeedOffset(m.noiseSeed);
        glm::vec2 dp =
            glm::vec2(pl.x, pl.z) - candleFlameCenter(h, seedP);
        float radial =
            glm::length(dp) / fmaxf(candleFlameRadius(h), 1e-4f);
        float hotBody =
            (1.0f - glm::smoothstep(0.38f, 0.92f, radial)) *
            glm::smoothstep(0.12f, 0.30f, h) *
            (1.0f - glm::smoothstep(0.72f, 0.96f, h));
        relativeTemperature =
            glm::clamp(0.64f + 0.26f * core + 0.10f * hotBody, 0.60f,
                       1.0f);
        source *= 1.08f - 0.20f * h;
    } else {
        relativeTemperature =
            glm::clamp(0.58f + 0.42f * core * (1.0f - 0.55f * h),
                       0.52f, 1.0f);
        source *= 1.15f - 0.42f * h;
    }
}

// --- Ray/medium interval ----------------------------------------------------

// Parametric interval [t0, t1] over which the (normalized, world-space) ray
// is inside medium geom g. Local t equals world t because the local direction
// is left unnormalized (worldPoint(t) and localPoint(t) share the parameter).
// Returns false if the ray misses the interior. CUBE and SPHERE only.
__host__ __device__ inline bool mediumInterval(const Geom &g, const Ray &r,
                                               float &t0, float &t1) {
    glm::vec3 ro =
        multiplyMV(g.transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd =
        glm::vec3(g.transform.inverseTransform * glm::vec4(r.direction, 0.0f));

    if (g.type == CUBE) {
        // Slab test against the local unit cube [-0.5, 0.5]^3.
        t0 = -FLT_MAX;
        t1 = FLT_MAX;
        for (int i = 0; i < 3; ++i) {
            if (fabsf(rd[i]) < 1e-9f) {
                if (ro[i] < -0.5f || ro[i] > 0.5f) {
                    return false;
                }
                continue;
            }
            float inv = 1.0f / rd[i];
            float ta = (-0.5f - ro[i]) * inv;
            float tb = (0.5f - ro[i]) * inv;
            t0 = fmaxf(t0, fminf(ta, tb));
            t1 = fminf(t1, fmaxf(ta, tb));
        }
        return t1 > fmaxf(t0, 0.0f);
    } else if (g.type == SPHERE) {
        // Local sphere of radius 0.5 centered at the origin.
        float a = glm::dot(rd, rd);
        if (a <= 0.0f) {
            return false;
        }
        float b = 2.0f * glm::dot(ro, rd);
        float c = glm::dot(ro, ro) - 0.25f;
        float disc = b * b - 4.0f * a * c;
        if (disc <= 0.0f) {
            return false;
        }
        float sq = sqrtf(disc);
        t0 = (-b - sq) / (2.0f * a);
        t1 = (-b + sq) / (2.0f * a);
        return t1 > fmaxf(t0, 0.0f);
    }
    return false; // MESH media unsupported (validated at scene load)
}

// --- Distance sampling ------------------------------------------------------

// Mean of a Spectrum's components: the Wilkie-style balance-heuristic pdf
// over the carried wavelengths (channels in RGB builds).
__host__ __device__ inline float avgComponent(const glm::vec3 &s) {
    return (s.x + s.y + s.z) * (1.0f / 3.0f);
}
__host__ __device__ inline float avgComponent(const glm::vec4 &s) {
    return (s.x + s.y + s.z + s.w) * 0.25f;
}

struct MediumSample {
    bool scattered;  // true: real scattering event at distance t
    float t;         // world distance of the event along the ray
    Spectrum weight; // throughput multiplier: sigma_s * T / pdf on a scatter,
                     // T / pdf (or the ratio-tracking product) on pass-through
    int nullCollisions;
    int brickVisits;
    int emptyBrickSkips;
    bool trackingOverflow;
};

// Homogeneous medium: analytic exponential sampling using the hero
// wavelength's sigma_t, balanced over all carried wavelengths. Purely
// absorbing media take a variance-free Beer-Lambert fast path: there can be
// no real scattering event, so sampling an absorption/escape Bernoulli event
// only adds noise and GPU work without changing the expectation.
__host__ __device__ inline MediumSample
sampleMediumHomogeneous(const Spectrum &sigT, const Spectrum &sigS, float tMax,
                        float u) {
    MediumSample ms;
    ms.scattered = false;
    ms.t = tMax;
    ms.weight = Spectrum(1.0f);
    ms.nullCollisions = 0;
    ms.brickVisits = 0;
    ms.emptyBrickSkips = 0;
    ms.trackingOverflow = false;

    if (maxComponent(sigS) <= 0.0f) {
        ms.weight = glm::exp(-sigT * tMax);
        return ms;
    }

    float sigTHero = sigT[0];
    if (sigTHero <= 0.0f) {
        return ms;
    }

    float t = -logf(fmaxf(1.0f - u, 1e-7f)) / sigTHero;
    if (t < tMax) {
        Spectrum Tr = glm::exp(-sigT * t);
        float pdf = avgComponent(sigT * Tr);
        if (pdf <= 0.0f) {
            return ms; // numerically opaque: treat as absorbed
        }
        ms.scattered = true;
        ms.t = t;
        ms.weight = sigS * Tr / pdf;
    } else {
        Spectrum Tr = glm::exp(-sigT * tMax);
        float pdf = avgComponent(Tr);
        if (pdf <= 0.0f) {
            ms.weight = Spectrum(0.0f);
            return ms;
        }
        ms.weight = Tr / pdf;
    }
    return ms;
}

// Heterogeneous medium: delta tracking against the constant majorant
// maxComponent(sigT), decisions driven by the hero wavelength, per-wavelength
// ratio weights at every null collision (PBRT-v4 style, unbiased).
__host__ __device__ inline MediumSample
sampleMediumHeterogeneous(const Geom &g, const Material &m,
                          const Spectrum &sigT, const Spectrum &sigS,
                          const Ray &ray, float tMax,
                          thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    MediumSample ms;
    ms.scattered = false;
    ms.t = tMax;
    ms.weight = Spectrum(1.0f);
    ms.nullCollisions = 0;
    ms.brickVisits = 0;
    ms.emptyBrickSkips = 0;
    ms.trackingOverflow = false;

    float sigMaj = maxComponent(sigT);
    if (sigMaj <= 0.0f) {
        return ms;
    }

    float t = 0.0f;
    // Bounded loop as a safety net; expected iteration count is
    // sigMaj * tMax.
    for (int i = 0; i < 10000; ++i) {
        t -= logf(fmaxf(1.0f - u01(rng), 1e-7f)) / sigMaj;
        if (t >= tMax) {
            break; // escaped: accumulated null weights are the answer
        }
        glm::vec3 p = ray.origin + t * ray.direction;
        float d = mediumDensity(m, g, p);
        Spectrum sigTx = d * sigT;
        float heroT = sigTx[0];
        if (u01(rng) * sigMaj < heroT) {
            // Real collision: scatter here. Per-wavelength vertex weight
            // sigma_s(x, lambda) / (hero collision pdf).
            ms.scattered = true;
            ms.t = t;
            ms.weight *= (d * sigS) / heroT;
            return ms;
        }
        // Null collision: per-wavelength ratio weight (1 for the hero lane).
        float heroNull = sigMaj - heroT;
        if (heroNull <= 0.0f) {
            break; // hero lane is at the majorant; null prob is 0
        }
        ms.weight *= (Spectrum(sigMaj) - sigTx) / heroNull;
    }
    return ms;
}

// Sparse combustion grid: piecewise delta tracking against a conservative
// majorant for the current 8^3 brick. Empty page-table entries jump directly
// to their next boundary. Pointwise extinction is reconstructed from the
// independently filtered smoke-density, soot, and reaction fields.
__device__ inline MediumSample sampleMediumSparseGrid(
    const Geom &g, const Spectrum &smokeA, const Spectrum &smokeS,
    const Spectrum &sootA, const Spectrum &sootS,
    const Spectrum &flameA, const Ray &ray, float tMax,
    thrust::default_random_engine &rng, VolumeTrackingStats *stats) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    MediumSample ms{};
    ms.scattered = false;
    ms.t = tMax;
    ms.weight = Spectrum(1.0f);

    const SparseVolumeGridDevice &grid = g.volumeGrid;
    if (!grid.valid || tMax <= 0.0f) {
        return ms;
    }

    const GridRay localRay = makeGridRay(g, ray);
    float t = 0.0f;
    unsigned long long visits = 0;
    unsigned long long emptySkips = 0;
    unsigned long long nulls = 0;
    unsigned long long violations = 0;
    const int maxBrickSteps = grid.brickResolution.x +
                              grid.brickResolution.y +
                              grid.brickResolution.z + 12;

    for (int step = 0; step < maxBrickSteps && t < tMax; ++step) {
        const GridBrickInterval interval =
            locateGridBrick(grid, localRay, t, tMax);
        ++visits;
        if (interval.tExit <= t) {
            ms.trackingOverflow = true;
            break;
        }
        if (interval.brickIndex < 0) {
            ++emptySkips;
            t = interval.tExit;
            continue;
        }

        const VolumeBrickMeta &meta = grid.bricks[interval.brickIndex];
        const float sigMaj =
            gridBrickMajorant(meta, smokeA, smokeS, sootA, sootS, flameA);
        if (sigMaj <= 0.0f) {
            t = interval.tExit;
            continue;
        }

        int candidateCount = 0;
        while (t < interval.tExit && candidateCount < 1000000) {
            ++candidateCount;
            t -= logf(fmaxf(1.0f - u01(rng), 1e-7f)) / sigMaj;
            if (t >= interval.tExit) {
                break;
            }
            CombustionFieldSample fields;
            const glm::vec3 pLocal =
                localRay.origin + t * localRay.direction;
            if (!sampleCombustionGridLocal(grid, pLocal, fields)) {
                ++nulls;
                continue;
            }
            const Spectrum sigAx =
                gridSigmaA(fields, smokeA, sootA, flameA);
            const Spectrum sigSx =
                gridSigmaS(fields, smokeS, sootS);
            const Spectrum sigTx = sigAx + sigSx;
            const float localMax = maxComponent(sigTx);
            if (localMax > sigMaj * (1.0f + 2e-4f) + 1e-6f) {
                ++violations;
            }
            const float heroT = sigTx[0];
            if (u01(rng) * sigMaj < heroT) {
                ms.scattered = true;
                ms.t = t;
                ms.weight *= sigSx / fmaxf(heroT, 1e-20f);
                ms.nullCollisions = static_cast<int>(nulls);
                ms.brickVisits = static_cast<int>(visits);
                ms.emptyBrickSkips = static_cast<int>(emptySkips);
                recordVolumeTrackingStats(stats, visits, emptySkips, nulls, 1,
                                          violations, 0);
                return ms;
            }
            const float heroNull = sigMaj - heroT;
            if (heroNull <= 0.0f) {
                ++violations;
                ms.trackingOverflow = true;
                break;
            }
            ms.weight *= (Spectrum(sigMaj) - sigTx) / heroNull;
            ++nulls;
        }
        if (candidateCount >= 1000000) {
            ms.trackingOverflow = true;
            break;
        }
        t = interval.tExit;
    }

    if (t < tMax) {
        ms.trackingOverflow = true;
    }
    ms.nullCollisions = static_cast<int>(nulls);
    ms.brickVisits = static_cast<int>(visits);
    ms.emptyBrickSkips = static_cast<int>(emptySkips);
    recordVolumeTrackingStats(stats, visits, emptySkips, nulls, 0, violations,
                              ms.trackingOverflow ? 1 : 0);
    if (ms.trackingOverflow) {
        // A loud, conservative failure is preferable to silently returning a
        // biased bright path. Normal validated grids never reach this branch.
        ms.weight = Spectrum(0.0f);
    }
    return ms;
}

// --- Transmittance (shadow rays) --------------------------------------------

// Transmittance of medium geom g over ray segment [t0, t1]. Analytic for
// homogeneous media; ratio tracking (Novak et al. 2014) for heterogeneous.
__host__ __device__ inline Spectrum
mediumTransmittance(const Geom &g, const Material &m, const Spectrum &sigT,
                    const Ray &ray, float t0, float t1,
                    thrust::default_random_engine &rng) {
    if (!m.heterogeneous) {
        return glm::exp(-sigT * (t1 - t0));
    }
    thrust::uniform_real_distribution<float> u01(0, 1);
    float sigMaj = maxComponent(sigT);
    if (sigMaj <= 0.0f) {
        return Spectrum(1.0f);
    }
    Spectrum Tr(1.0f);
    float t = t0;
    for (int i = 0; i < 10000; ++i) {
        t -= logf(fmaxf(1.0f - u01(rng), 1e-7f)) / sigMaj;
        if (t >= t1) {
            break;
        }
        glm::vec3 p = ray.origin + t * ray.direction;
        float d = mediumDensity(m, g, p);
        Tr *= (Spectrum(sigMaj) - d * sigT) / sigMaj;
        // Russian roulette on a nearly opaque channel product to bound the
        // loop in dense media (PBRT-v4 does the same).
        float maxTr = maxComponent(Tr);
        if (maxTr < 0.05f) {
            float q = 0.75f;
            if (u01(rng) < q) {
                return Spectrum(0.0f);
            }
            Tr /= 1.0f - q;
        }
    }
    return Tr;
}

// Ratio-tracked transmittance through a sparse field, using the same local
// majorants and DDA empty-space skipping as free-flight sampling.
__device__ inline Spectrum sparseGridTransmittance(
    const Geom &g, const Spectrum &smokeA, const Spectrum &smokeS,
    const Spectrum &sootA, const Spectrum &sootS,
    const Spectrum &flameA, const Ray &ray, float t0, float t1,
    thrust::default_random_engine &rng, VolumeTrackingStats *stats) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    const SparseVolumeGridDevice &grid = g.volumeGrid;
    if (!grid.valid || t1 <= t0) {
        return Spectrum(1.0f);
    }

    const GridRay localRay = makeGridRay(g, ray);
    Spectrum Tr(1.0f);
    float t = t0;
    unsigned long long visits = 0;
    unsigned long long emptySkips = 0;
    unsigned long long nulls = 0;
    unsigned long long violations = 0;
    bool overflow = false;
    const int maxBrickSteps = grid.brickResolution.x +
                              grid.brickResolution.y +
                              grid.brickResolution.z + 12;

    for (int step = 0; step < maxBrickSteps && t < t1; ++step) {
        const GridBrickInterval interval =
            locateGridBrick(grid, localRay, t, t1);
        ++visits;
        if (interval.tExit <= t) {
            overflow = true;
            break;
        }
        if (interval.brickIndex < 0) {
            ++emptySkips;
            t = interval.tExit;
            continue;
        }

        const VolumeBrickMeta &meta = grid.bricks[interval.brickIndex];
        const float sigMaj =
            gridBrickMajorant(meta, smokeA, smokeS, sootA, sootS, flameA);
        if (sigMaj <= 0.0f) {
            t = interval.tExit;
            continue;
        }

        int candidateCount = 0;
        while (t < interval.tExit && candidateCount < 1000000) {
            ++candidateCount;
            t -= logf(fmaxf(1.0f - u01(rng), 1e-7f)) / sigMaj;
            if (t >= interval.tExit) {
                break;
            }
            CombustionFieldSample fields;
            const glm::vec3 pLocal =
                localRay.origin + t * localRay.direction;
            Spectrum sigTx(0.0f);
            if (sampleCombustionGridLocal(grid, pLocal, fields)) {
                sigTx =
                    gridSigmaA(fields, smokeA, sootA, flameA) +
                    gridSigmaS(fields, smokeS, sootS);
            }
            if (maxComponent(sigTx) >
                sigMaj * (1.0f + 2e-4f) + 1e-6f) {
                ++violations;
            }
            Tr *= (Spectrum(sigMaj) - sigTx) / sigMaj;
            ++nulls;

            const float maxTr = maxComponent(Tr);
            if (maxTr < 0.05f) {
                constexpr float q = 0.75f;
                if (u01(rng) < q) {
                    recordVolumeTrackingStats(stats, visits, emptySkips,
                                              nulls, 0, violations, 0);
                    return Spectrum(0.0f);
                }
                Tr /= 1.0f - q;
            }
        }
        if (candidateCount >= 1000000) {
            overflow = true;
            break;
        }
        t = interval.tExit;
    }

    if (t < t1) {
        overflow = true;
    }
    recordVolumeTrackingStats(stats, visits, emptySkips, nulls, 0, violations,
                              overflow ? 1 : 0);
    return overflow ? Spectrum(0.0f) : Tr;
}
