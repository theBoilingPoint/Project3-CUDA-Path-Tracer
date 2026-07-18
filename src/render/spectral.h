#pragma once

// ============================================================================
// Spectral rendering (hero-wavelength) toggle and shared types.
//
// SPECTRAL 0: the renderer transports RGB triples end-to-end (original
//             behavior, bit-identical).
// SPECTRAL 1: the renderer transports radiance at NSpectrumSamples sampled
//             wavelengths per path (hero-wavelength spectral sampling,
//             Wilkie et al. 2014 / PBRT-v4 style). RGB assets are uplifted to
//             smooth spectra (Jakob & Hanika 2019) inside the shade kernel and
//             converted back to RGB through the CIE 1931 color-matching
//             functions in finalGather. Enables dispersion (see Material.abbe)
//             and physically meaningful illuminant spectra.
//
// This header is the single cross-file switch: it is included via
// sceneStructs.h so every translation unit (CUDA and host) agrees on the
// Spectrum type and the PathSegment layout. The per-feature toggles local to
// pathtrace.cu (USE_MIS etc.) stay where they are.
// ============================================================================
#define SPECTRAL 1

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <cmath>

// Number of wavelengths carried per path. Fixed at 4: Spectrum is glm::vec4
// and several loops below rely on it. (PBRT-v4's default as well.)
constexpr int NSpectrumSamples = 4;

// Visible range covered by the wavelength sampler and the CIE tables, in nm.
constexpr float LAMBDA_MIN = 360.0f;
constexpr float LAMBDA_MAX = 830.0f;

// Integral of the CIE Y matching curve over [360, 830] nm. Dividing XYZ
// estimates by this makes a spectrum equal to the CIE Y curve have luminance 1.
constexpr float CIE_Y_INTEGRAL = 106.856895f;

#if SPECTRAL
// Radiance/throughput/reflectance sampled at the path's 4 wavelengths.
using Spectrum = glm::vec4;
#else
// RGB mode: plain RGB triple, exactly as before.
using Spectrum = glm::vec3;
#endif

#if SPECTRAL
// The 4 wavelengths a path transports, plus the pdf each was sampled with.
// lambda[0] is the hero wavelength: all directional sampling decisions are
// based on it alone. Carried in PathSegment from camera ray to finalGather.
struct SampledWavelengths {
    glm::vec4 lambda; // nm; lambda[0] = hero
    glm::vec4 pdf;    // sampling pdf per wavelength (1/nm)

    __host__ __device__ bool secondaryTerminated() const {
        return pdf[1] == 0.0f && pdf[2] == 0.0f && pdf[3] == 0.0f;
    }

    // Collapse the path to its hero wavelength only. Called when the path
    // crosses a wavelength-dependent (dispersive) interface, where one
    // refracted direction can only be correct for one wavelength.
    // PBRT-v4 semantics: zero the secondary pdfs and divide the hero pdf by
    // NSpectrumSamples -- this cancels the 1/N average in the sensor estimate
    // so the surviving hero-only estimate stays unbiased.
    __host__ __device__ void terminateSecondary() {
        if (secondaryTerminated()) {
            return;
        }
        pdf[1] = pdf[2] = pdf[3] = 0.0f;
        pdf[0] /= (float)NSpectrumSamples;
    }
};
#else
// RGB mode: empty tag type so function signatures never fork on the macro.
struct SampledWavelengths {};
#endif

#if SPECTRAL
// --- Visible-wavelength importance sampling -------------------------------
// pdf proportional to a squared hyperbolic secant fit of the CIE luminous
// response (Radziszewski et al. 2009; constants from PBRT-v4's
// SampleVisibleWavelengths). Concentrates samples where the eye is sensitive
// instead of wasting them on the deep violet/red tails.

__host__ __device__ inline float visibleWavelengthsPDF(float lambda) {
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) {
        return 0.0f;
    }
    float ch = coshf(0.0072f * (lambda - 538.0f));
    return 0.0039398042f / (ch * ch);
}

__host__ __device__ inline float sampleVisibleWavelength(float u) {
    return 538.0f - 138.888889f * atanhf(0.85691062f - 1.82750197f * u);
}

// Sample the path's 4 wavelengths from one uniform draw: the hero wavelength
// from u itself and the other three from rotations of u by i/4 (stratified
// over the visible band, PBRT SampledWavelengths::SampleVisible).
__host__ __device__ inline SampledWavelengths sampleWavelengths(float u) {
    SampledWavelengths swl;
    for (int i = 0; i < NSpectrumSamples; ++i) {
        float up = u + (float)i / (float)NSpectrumSamples;
        if (up > 1.0f) {
            up -= 1.0f;
        }
        swl.lambda[i] = sampleVisibleWavelength(up);
        swl.pdf[i] = visibleWavelengthsPDF(swl.lambda[i]);
    }
    return swl;
}
#endif // SPECTRAL

// --- Dispersion ------------------------------------------------------------
// Wavelength-dependent index of refraction from a two-term Cauchy model
// n(lambda) = B + C / lambda^2, with B and C derived from the glass's d-line
// IOR (Material.indexOfRefraction, at 587.6 nm) and its Abbe number
// V = (n_d - 1) / (n_F - n_C) using the F (486.1 nm) and C (656.3 nm) lines.
// Reference values: BK7 crown V=64.17, SF11 flint V=25.68, diamond V=55.3.
__host__ __device__ inline float cauchyIOR(float nd, float abbe,
                                           float lambdaNm) {
    // Fraunhofer line wavelengths, in micrometers.
    constexpr float lF = 0.4861f, lC = 0.6563f, lD = 0.5876f;
    float C = (nd - 1.0f) / (abbe * (1.0f / (lF * lF) - 1.0f / (lC * lC)));
    float B = nd - C / (lD * lD);
    float lUm = lambdaNm * 1e-3f;
    return B + C / (lUm * lUm);
}

// Named emission spectra selectable per light in the scene JSON ("SPECTRUM").
// Only meaningful in SPECTRAL builds; RGB builds fold the illuminant's RGB
// equivalent into the light color at scene load instead.
enum SpectrumType {
    SPECTRUM_NONE = 0,  // RGB uplift only (scaled sigmoid x D65, PBRT-style)
    SPECTRUM_D65,       // CIE standard daylight, 6504 K
    SPECTRUM_A,         // CIE incandescent/tungsten, 2856 K
    SPECTRUM_E,         // equal-energy white
    SPECTRUM_BLACKBODY, // Planck radiator at Material.blackbodyTemp kelvin
};

// --- Blackbody emission ------------------------------------------------------
// Planck's law: spectral radiance of a blackbody at temperature T (kelvin) at
// wavelength lambda (nm). Absolute units cancel out -- emitters multiply by a
// host-computed normalization that gives the SPD unit luminance.
__host__ __device__ inline float planckSPD(float lambdaNm, float T) {
    if (T <= 0.0f) {
        return 0.0f;
    }
    const float c = 299792458.0f;    // m/s
    const float h = 6.62606957e-34f; // J s
    const float kb = 1.3806488e-23f; // J/K
    float l = lambdaNm * 1e-9f;
    float l2 = l * l;
    float l5 = l2 * l2 * l;
    return (2.0f * h * c * c) / (l5 * (expf((h * c) / (l * kb * T)) - 1.0f));
}

// --- Small mode-agnostic helpers so shading code has no #if forks ----------
__host__ __device__ inline float maxComponent(const glm::vec3 &s) {
    return fmaxf(s.x, fmaxf(s.y, s.z));
}

__host__ __device__ inline float maxComponent(const glm::vec4 &s) {
    return fmaxf(fmaxf(s.x, s.y), fmaxf(s.z, s.w));
}

__host__ __device__ inline bool isBlack(const glm::vec3 &s) {
    return s.x <= 0.0f && s.y <= 0.0f && s.z <= 0.0f;
}

__host__ __device__ inline bool isBlack(const glm::vec4 &s) {
    return s.x <= 0.0f && s.y <= 0.0f && s.z <= 0.0f && s.w <= 0.0f;
}
