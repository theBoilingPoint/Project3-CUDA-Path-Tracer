#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "spectral.h"

// ---------------------------------------------------------------------------
// Host helpers available in BOTH modes.
// ---------------------------------------------------------------------------

// Linear-sRGB color of a unit-luminance named illuminant (SpectrumType).
// RGB builds use this at scene load to fold a SPECTRUM-tagged light's color
// into its RGB tint -- a graceful approximation of the spectral behavior.
glm::vec3 illuminantRGB(int spectrumType, float blackbodyTemp);

// Scale that normalizes the Planck SPD at tempK to unit luminance under the
// renderer's sampling convention. Stored in Material/DeltaLight::blackbodyNorm
// at scene load so the device evaluates blackbody emission analytically.
float blackbodyLuminanceNorm(float tempK);

#if SPECTRAL

// Number of 1 nm bins covering [LAMBDA_MIN, LAMBDA_MAX] = [360, 830] nm.
constexpr int N_CIE_BINS = 471;

// Device-side spectral tables, filled once at startup by initSpectralTables().
// The fixed-size tables live in __constant__ memory (~9.4 KB of the 64 KB
// budget); the rgb2spec coefficient table is far too large for that (~9.4 MB)
// and lives in global memory, pointed to from here.
struct SpectralTablesDev {
    // CIE 1931 2-degree color matching functions, 1 nm bins.
    float cieX[N_CIE_BINS];
    float cieY[N_CIE_BINS];
    float cieZ[N_CIE_BINS];
    // Standard illuminants, 1 nm bins, normalized to unit luminance (i.e.
    // sum(illum[bin] * cieY[bin]) == CIE_Y_INTEGRAL) so an emitter's
    // EMITTANCE keeps controlling perceived brightness across illuminants.
    float d65[N_CIE_BINS];
    float illumA[N_CIE_BINS];
    // rgb2spec sRGB model (Jakob & Hanika 2019): scale[res] and
    // data[3 * res^3 * RGB2SPEC_N_COEFFS], both device-global.
    const float *rgb2specData;
    const float *rgb2specScale;
    int rgb2specRes;
};

extern __constant__ SpectralTablesDev c_spectral;

// Host: load srgb.coeff (generated at build time, placed next to the
// executable), build the 1 nm CIE tables and upload everything to the GPU.
// Also self-checks the GPU port below against the reference C implementation.
// Exits with a message if the coefficient table is missing. Idempotent.
void initSpectralTables();
// Host: release the global-memory rgb2spec buffers (no-op if not loaded).
void freeSpectralTables();

// ---------------------------------------------------------------------------
// Portable port of rgb2spec_fetch()/rgb2spec_eval_precise() from
// libs/rgb2spec/rgb2spec.c, usable from both host (unit-checked at startup
// against the reference implementation) and device (called by the uplift
// functions in pathtrace.cu with the c_spectral pointers). Mirrors the
// reference logic exactly -- keep the two in sync.
// ---------------------------------------------------------------------------

// Binary search for the interval of `scaleArr` containing x (rgb2spec_find_interval).
__host__ __device__ inline int rs_findInterval(const float *scaleArr, int size_,
                                               float x) {
    int left = 0, lastInterval = size_ - 2, size = lastInterval;
    while (size > 0) {
        int half = size >> 1, middle = left + half + 1;
        if (scaleArr[middle] <= x) {
            left = middle;
            size -= half + 1;
        } else {
            size = half;
        }
    }
    return left < lastInterval ? left : lastInterval;
}

// RGB in [0,1] -> 3 sigmoid-polynomial coefficients (rgb2spec_fetch).
__host__ __device__ inline void rs_fetchCoeffs(const float *data,
                                               const float *scaleArr, int res,
                                               const glm::vec3 &rgbIn,
                                               float out[3]) {
    float rgb[3];
    rgb[0] = fminf(fmaxf(rgbIn.x, 0.0f), 1.0f);
    rgb[1] = fminf(fmaxf(rgbIn.y, 0.0f), 1.0f);
    rgb[2] = fminf(fmaxf(rgbIn.z, 0.0f), 1.0f);

    // Closed-form solution for monochromatic inputs (rgb2spec_fetch_mono):
    // black/white map to -/+8192, which the sigmoid evaluates to exactly 0/1.
    if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
        float v = rgb[0], r;
        if (v <= 0.0f) {
            r = -8192.0f;
        } else if (v >= 1.0f) {
            r = 8192.0f;
        } else {
            r = (v - 0.5f) / sqrtf(v * (1.0f - v));
        }
        out[0] = out[1] = 0.0f;
        out[2] = r;
        return;
    }

    // Largest component picks the table partition (ties: last one wins,
    // matching the reference's `>=`).
    int i = 0;
    for (int j = 1; j < 3; ++j) {
        if (rgb[j] >= rgb[i]) {
            i = j;
        }
    }

    float z = rgb[i];
    if (z <= 0.0f) {
        out[0] = data[0];
        out[1] = data[1];
        out[2] = data[2];
        return;
    }

    float scale = (float)(res - 1) / z;
    float x = rgb[(i + 1) % 3] * scale;
    float y = rgb[(i + 2) % 3] * scale;

    unsigned int xi = (unsigned int)x, yi = (unsigned int)y;
    if (xi > (unsigned int)(res - 2)) xi = (unsigned int)(res - 2);
    if (yi > (unsigned int)(res - 2)) yi = (unsigned int)(res - 2);
    unsigned int zi = (unsigned int)rs_findInterval(scaleArr, res, z);
    unsigned int offset =
        (((i * res + zi) * res + yi) * res + xi) * 3;
    unsigned int dx = 3, dy = 3 * res, dz = 3 * res * res;

    float x1 = x - xi, x0 = 1.0f - x1;
    float y1 = y - yi, y0 = 1.0f - y1;
    float z1 = (z - scaleArr[zi]) / (scaleArr[zi + 1] - scaleArr[zi]);
    float z0 = 1.0f - z1;

    for (int j = 0; j < 3; ++j) {
        out[j] = ((data[offset] * x0 + data[offset + dx] * x1) * y0 +
                  (data[offset + dy] * x0 + data[offset + dy + dx] * x1) * y1) *
                     z0 +
                 ((data[offset + dz] * x0 + data[offset + dz + dx] * x1) * y0 +
                  (data[offset + dz + dy] * x0 +
                   data[offset + dz + dy + dx] * x1) *
                      y1) *
                     z1;
        offset++;
    }
}

// Evaluate the sigmoid polynomial at wavelength lambda in nm
// (rgb2spec_eval_precise).
__host__ __device__ inline float rs_evalSigmoid(const float c[3],
                                                float lambda) {
    float x = fmaf(fmaf(c[0], lambda, c[1]), lambda, c[2]);
    float y = 1.0f / sqrtf(fmaf(x, x, 1.0f));
    return fmaf(0.5f * x, y, 0.5f);
}

#endif // SPECTRAL
