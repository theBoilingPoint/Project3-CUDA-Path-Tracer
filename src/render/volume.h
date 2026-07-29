#pragma once

#include <cfloat>
#include <glm/glm.hpp>
#include <thrust/random.h>

#include "intersections.h" // multiplyMV, utilhash
#include "sceneStructs.h"  // Spectrum (via spectral.h), Geom, Material, Ray

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

    if (g.type == SPHERE) {
        float r = glm::length(pl);
        d *= glm::clamp((0.5f - r) / 0.10f, 0.0f, 1.0f);
    } else { // CUBE
        glm::vec3 a = glm::vec3(0.5f) - glm::abs(pl);
        float edge = fminf(a.x, fminf(a.y, a.z));
        float edgeWidth = m.mediumProfile == MEDIUM_PROFILE_FOG
                              ? 0.13f
                              : (m.mediumProfile == MEDIUM_PROFILE_CLOUD
                                     ? 0.10f
                                     : 0.04f);
        d *= glm::clamp(edge / edgeWidth, 0.0f, 1.0f);
    }
    return d;
}

// Spatial source strength and relative temperature for continuous medium
// emission. Fire is hottest at its dense lower core and cooler toward its
// thin upper/outer tongues. Other emissive profiles simply follow density.
__host__ __device__ inline void
mediumEmissionProperties(const Material &m, const Geom &g,
                         const glm::vec3 &pWorld, float &source,
                         float &relativeTemperature) {
    source = mediumDensity(m, g, pWorld);
    relativeTemperature = 1.0f;
    if (m.mediumProfile != MEDIUM_PROFILE_FLAME || source <= 0.0f) {
        return;
    }

    glm::vec3 pl =
        multiplyMV(g.transform.inverseTransform, glm::vec4(pWorld, 1.0f));
    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);
    float core = sqrtf(glm::clamp(source, 0.0f, 1.0f));
    relativeTemperature =
        glm::clamp(0.58f + 0.42f * core * (1.0f - 0.55f * h), 0.52f, 1.0f);
    source *= 1.15f - 0.42f * h;
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
};

// Homogeneous medium: analytic exponential sampling using the hero
// wavelength's sigma_t, balanced over all carried wavelengths.
__host__ __device__ inline MediumSample
sampleMediumHomogeneous(const Spectrum &sigT, const Spectrum &sigS, float tMax,
                        float u) {
    MediumSample ms;
    ms.scattered = false;
    ms.t = tMax;
    ms.weight = Spectrum(1.0f);

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
