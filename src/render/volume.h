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
    return (1.0f + g * g - s * s) / (2.0f * g);
}

// --- Procedural density field (heterogeneous media) -------------------------

// Deterministic lattice hash -> [0, 1].
__host__ __device__ inline float noiseHash(int x, int y, int z) {
    unsigned int h = (unsigned int)(x * 73856093) ^
                     (unsigned int)(y * 19349663) ^
                     (unsigned int)(z * 83492791);
    return (float)(utilhash(h) & 0x00FFFFFFu) / 16777215.0f;
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

// Cumulus profile: distinct large lobes (kept unmerged so the valleys
// between them survive and self-shadow) plus smaller towers on top, with
// inverted-Worley billow erosion applied as a REMAP through the whole
// volume -- interior structure, not just a decorated boundary.
__host__ __device__ inline float cloudDensity(const glm::vec3 &pl,
                                              float noiseScale, int octaves) {
    // Large lobes along the long axis.
    float base = 0.0f;
    for (int i = 0; i < 6; ++i) {
        glm::vec3 c((noiseHash(i, 1, 7) - 0.5f) * 0.62f,
                    noiseHash(i, 3, 11) * 0.16f - 0.10f,
                    (noiseHash(i, 5, 13) - 0.5f) * 0.30f);
        float r = 0.16f + 0.20f * noiseHash(i, 7, 17);
        glm::vec3 d = pl - c;
        d.y /= 0.85f;
        base = fmaxf(base, 1.0f - glm::length(d) / r);
    }
    // Smaller cauliflower towers rising off the top.
    for (int i = 6; i < 10; ++i) {
        glm::vec3 c((noiseHash(i, 1, 7) - 0.5f) * 0.55f,
                    0.06f + noiseHash(i, 3, 11) * 0.20f,
                    (noiseHash(i, 5, 13) - 0.5f) * 0.26f);
        float r = 0.10f + 0.12f * noiseHash(i, 7, 17);
        base = fmaxf(base, 1.0f - glm::length(pl - c) / r);
    }
    base = glm::clamp(base * 1.25f, 0.0f, 1.0f);
    // Flat cloud base: real cumulus condense above a sharp altitude line.
    base *= glm::clamp((pl.y + 0.26f) / 0.08f, 0.0f, 1.0f);
    if (base <= 0.0f) {
        return 0.0f;
    }

    // Domain-warped billow (inverted Worley) + fine value-noise detail,
    // combined into an erosion threshold and REMAPPED: carves florets into
    // the interior and crenellates the silhouette.
    glm::vec3 q = pl * noiseScale;
    glm::vec3 warp(valueNoise(q * 0.35f + glm::vec3(13.1f)),
                   valueNoise(q * 0.35f + glm::vec3(47.7f)),
                   valueNoise(q * 0.35f + glm::vec3(91.3f)));
    float billow = billowFbm(q * 0.5f + 0.9f * (warp - glm::vec3(0.5f)));
    float det = fbm(q * 1.7f, octaves > 4 ? octaves - 2 : 2);
    float ero = (1.0f - billow) * 0.62f + (1.0f - det) * 0.13f;
    return glm::clamp((base - ero) / fmaxf(1.0f - ero, 1e-3f), 0.0f, 1.0f);
}

// Rising-plume profile: a stack of puff balls along a wandering, gently
// spiraling rise path -- small and tight at the base, swelling into a
// mushrooming head -- eroded by Worley billows that strengthen with height.
// Fluid-sim plumes are essentially stacked vortex rings, which this mimics;
// a noised cone reads as a funnel no matter how it is decorated.
__host__ __device__ inline float plumeDensity(const glm::vec3 &pl,
                                              float noiseScale, int octaves) {
    float base = 0.0f;
    for (int k = 0; k < 16; ++k) {
        float t = (float)k / 15.0f;
        // Wandering axis + gentle spiral, growing with height.
        float wx =
            (fbm(glm::vec3(t * 2.1f + 9.7f, 3.1f, 6.2f), 2) - 0.5f) * 0.55f * t;
        float wz =
            (fbm(glm::vec3(t * 2.1f + 41.3f, 8.4f, 2.6f), 2) - 0.5f) * 0.55f *
            t;
        float sp = 0.05f + 0.10f * t;
        float ang = 6.28318f * (noiseHash(k, 21, 5) + 1.6f * t);
        glm::vec3 c(wx + sp * cosf(ang), -0.5f + 0.88f * powf(t, 0.9f),
                    wz + sp * sinf(ang));
        // Puffs swell toward the head, with per-puff size jitter; the floor
        // keeps consecutive stem puffs overlapping (no gaps in the column).
        float r = (0.075f + 0.30f * powf(t, 1.4f)) *
                  (0.8f + 0.4f * noiseHash(k, 9, 33));
        float cov = 1.0f - glm::length(pl - c) / fmaxf(r, 1e-4f);
        base = fmaxf(base, cov);
    }
    if (base <= 0.0f) {
        return 0.0f;
    }
    base = glm::clamp(base * 1.35f, 0.0f, 1.0f);

    float h = glm::clamp(pl.y + 0.5f, 0.0f, 1.0f);

    // Worley billow + fine detail erosion, stronger with height: the stem
    // stays near-solid, the head breaks into lobes and wisps.
    glm::vec3 q = pl * noiseScale;
    float billow = billowFbm(q * 0.7f);
    float det = fbm(q * 1.9f + glm::vec3(31.4f), 3);
    float eroStr = 0.25f + 0.55f * h;
    float ero = ((1.0f - billow) * 0.8f + (1.0f - det) * 0.2f) * eroStr;
    float d = (base - ero) / fmaxf(1.0f - ero, 1e-3f);

    // Fade in at the very bottom; soft cap so the head never clips the geom.
    d *= glm::clamp(h / 0.05f, 0.0f, 1.0f);
    d *= 1.0f - glm::smoothstep(0.85f, 1.0f, h);
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
        d = cloudDensity(pl, m.noiseScale, m.noiseOctaves);
        break;
    case MEDIUM_PROFILE_PLUME:
        d = plumeDensity(pl, m.noiseScale, m.noiseOctaves);
        break;
    case MEDIUM_PROFILE_FBM:
    default:
        // Thresholded fbm: carve empty pockets, rescale the rest to [0, 1].
        d = fbm(pl * m.noiseScale, m.noiseOctaves);
        d = glm::clamp((d - 0.42f) / 0.33f, 0.0f, 1.0f);
        break;
    }

    if (g.type == SPHERE) {
        float r = glm::length(pl);
        d *= glm::clamp((0.5f - r) / 0.10f, 0.0f, 1.0f);
    } else { // CUBE
        glm::vec3 a = glm::vec3(0.5f) - glm::abs(pl);
        float edge = fminf(a.x, fminf(a.y, a.z));
        d *= glm::clamp(edge / 0.04f, 0.0f, 1.0f);
    }
    return d;
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
