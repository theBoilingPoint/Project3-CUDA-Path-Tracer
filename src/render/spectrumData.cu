#include "spectrumData.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

#include "cudaUtil.h"
#include "rgb2spec/rgb2spec.h"
// 5 nm CIE 1931 curves, D65 (unit luminance) and XYZ<->sRGB matrices shipped
// with rgb2spec. Only THIS translation unit may include cie1931.h: it defines
// a non-inline function (cie_interp), so a second inclusion would violate ODR.
#include "rgb2spec/details/cie1931.h"

// Number of 1 nm bins covering [LAMBDA_MIN, LAMBDA_MAX] (a local mirror of
// N_CIE_BINS, which spectrumData.h only defines in SPECTRAL builds).
static const int kBins = (int)(LAMBDA_MAX - LAMBDA_MIN) + 1;

// CIE standard illuminant A: a Planck radiator at 2856 K with the current
// (ITS-90) second radiation constant c2 = 1.4388e7 nm*K. Relative SPD; the
// caller normalizes to unit luminance, so the "100 at 560nm" convention
// drops out.
static double illuminantASPD(double lambdaNm) {
    const double c2 = 1.4388e7; // nm * K
    const double T = 2856.0;
    return 100.0 * std::pow(560.0 / lambdaNm, 5.0) *
           (std::exp(c2 / (T * 560.0)) - 1.0) /
           (std::exp(c2 / (T * lambdaNm)) - 1.0);
}

// Fill spd[kBins] with the (relative) SPD of the given SpectrumType at 1 nm.
static void buildIlluminantSPD(int spectrumType, float blackbodyTemp,
                               std::vector<double> &spd) {
    spd.resize(kBins);
    for (int bin = 0; bin < kBins; ++bin) {
        double lambda = (double)LAMBDA_MIN + bin;
        switch (spectrumType) {
            case SPECTRUM_A:
                spd[bin] = illuminantASPD(lambda);
                break;
            case SPECTRUM_E:
                spd[bin] = 1.0;
                break;
            case SPECTRUM_BLACKBODY:
                spd[bin] = (double)planckSPD((float)lambda, blackbodyTemp);
                break;
            case SPECTRUM_NONE:
            case SPECTRUM_D65:
            default:
                spd[bin] = cie_interp(cie_d65, lambda);
                break;
        }
    }
}

// Luminance of a 1 nm SPD under the renderer's convention:
// Y = sum(spd * cieY) / CIE_Y_INTEGRAL.
static double spdLuminance(const std::vector<double> &spd) {
    double lum = 0.0;
    for (int bin = 0; bin < kBins; ++bin) {
        double lambda = (double)LAMBDA_MIN + bin;
        lum += spd[bin] * cie_interp(cie_y, lambda);
    }
    return lum / CIE_Y_INTEGRAL;
}

float blackbodyLuminanceNorm(float tempK) {
    std::vector<double> spd;
    buildIlluminantSPD(SPECTRUM_BLACKBODY, tempK, spd);
    double lum = spdLuminance(spd);
    return lum > 0.0 ? (float)(1.0 / lum) : 0.0f;
}

glm::vec3 illuminantRGB(int spectrumType, float blackbodyTemp) {
    if (spectrumType == SPECTRUM_NONE) {
        // Plain RGB uplift is defined against D65 (the sRGB whitepoint), which
        // round-trips to white -- no tint.
        return glm::vec3(1.0f);
    }
    std::vector<double> spd;
    buildIlluminantSPD(spectrumType, blackbodyTemp, spd);
    double lum = spdLuminance(spd);
    if (lum <= 0.0) {
        return glm::vec3(1.0f);
    }
    // Unit-luminance SPD -> XYZ -> linear sRGB.
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (int bin = 0; bin < kBins; ++bin) {
        double lambda = (double)LAMBDA_MIN + bin;
        double s = spd[bin] / lum;
        X += s * cie_interp(cie_x, lambda);
        Y += s * cie_interp(cie_y, lambda);
        Z += s * cie_interp(cie_z, lambda);
    }
    X /= CIE_Y_INTEGRAL;
    Y /= CIE_Y_INTEGRAL;
    Z /= CIE_Y_INTEGRAL;
    glm::vec3 rgb(0.0f);
    for (int r = 0; r < 3; ++r) {
        rgb[r] = (float)(xyz_to_srgb[r][0] * X + xyz_to_srgb[r][1] * Y +
                         xyz_to_srgb[r][2] * Z);
    }
    // A physical illuminant can sit slightly outside the sRGB gamut (deep
    // incandescent reds); clamp so the RGB fallback stays a valid tint.
    return glm::max(rgb, glm::vec3(0.0f));
}

#if SPECTRAL

__constant__ SpectralTablesDev c_spectral;

static float *s_rgb2specData = nullptr;
static float *s_rgb2specScale = nullptr;
static bool s_loaded = false;

// Probe kernel: read back a few values of c_spectral THROUGH DEVICE CODE (not
// cudaMemcpyFromSymbol) so we verify exactly what the render kernels will see.
// Guards against the constant upload silently not reaching the linked symbol.
__global__ void probeSpectralTables(float *out) {
    out[0] = c_spectral.cieY[538 - (int)LAMBDA_MIN]; // ~0.88 at 538 nm
    out[1] = c_spectral.d65[540 - (int)LAMBDA_MIN];  // > 0
    out[2] = (float)c_spectral.rgb2specRes;          // 64
    out[3] = c_spectral.rgb2specScale != nullptr ? c_spectral.rgb2specScale[0]
                                                 : -1.0f;
}

// Directory containing the running executable; srgb.coeff is emitted there by
// the build (see the rgb2spec block in CMakeLists.txt), so loading is
// independent of the current working directory.
static std::string executableDir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, buf, MAX_PATH);
    std::string path(buf, n);
    size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    buf[n] = '\0';
    std::string path(buf);
    size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
#endif
}

// Verify the header-inlined GPU port (rs_fetchCoeffs/rs_evalSigmoid) against
// the reference C implementation on a few hundred random (rgb, lambda) pairs.
// The port runs on the HOST here, but it is the exact code the device runs.
static void selfCheckAgainstReference(RGB2Spec *model) {
    std::mt19937 gen(0x5eed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float maxErr = 0.0f;
    for (int k = 0; k < 500; ++k) {
        glm::vec3 rgb(u01(gen), u01(gen), u01(gen));
        float lambda = LAMBDA_MIN + u01(gen) * (LAMBDA_MAX - LAMBDA_MIN);

        float ref[RGB2SPEC_N_COEFFS], port[3];
        float rgbArr[3] = {rgb.x, rgb.y, rgb.z};
        rgb2spec_fetch(model, rgbArr, ref);
        rs_fetchCoeffs(model->data, model->scale, (int)model->res, rgb, port);

        float refV = rgb2spec_eval_precise(ref, lambda);
        float portV = rs_evalSigmoid(port, lambda);
        float err = std::fabs(refV - portV);
        if (err > maxErr) {
            maxErr = err;
        }
    }
    if (maxErr > 1e-4f) {
        fprintf(stderr,
                "FATAL: rgb2spec GPU port disagrees with the reference "
                "implementation (max |err| = %g). The port in spectrumData.h "
                "is out of sync with libs/rgb2spec/rgb2spec.c.\n",
                maxErr);
        exit(EXIT_FAILURE);
    }
    printf("rgb2spec self-check passed (max |err| = %g over 500 samples)\n",
           maxErr);
}

void initSpectralTables() {
    if (s_loaded) {
        return;
    }

    // --- rgb2spec sRGB coefficient table --------------------------------
    std::string path = executableDir() + "/srgb.coeff";
    RGB2Spec *model = rgb2spec_load(path.c_str());
    if (!model) {
        fprintf(stderr,
                "FATAL: SPECTRAL build could not load the rgb2spec table at\n"
                "  %s\n"
                "It is generated at build time by the rgb2spec_opt target; "
                "rebuild the project (or run: rgb2spec_opt 64 srgb.coeff sRGB "
                "next to the executable).\n",
                path.c_str());
        exit(EXIT_FAILURE);
    }

    selfCheckAgainstReference(model);

    int res = (int)model->res;
    size_t sizeScale = sizeof(float) * res;
    size_t sizeData =
        sizeof(float) * (size_t)res * res * res * 3 * RGB2SPEC_N_COEFFS;
    cudaMalloc(&s_rgb2specScale, sizeScale);
    cudaMalloc(&s_rgb2specData, sizeData);
    cudaMemcpy(s_rgb2specScale, model->scale, sizeScale,
               cudaMemcpyHostToDevice);
    cudaMemcpy(s_rgb2specData, model->data, sizeData, cudaMemcpyHostToDevice);

    // --- 1 nm CIE tables --------------------------------------------------
    // Resampled from the 5 nm data in cie1931.h with linear interpolation
    // (cie_interp), matching how PBRT builds its densely sampled spectra.
    static SpectralTablesDev host; // ~9.4 KB; static to keep it off the stack
    for (int bin = 0; bin < N_CIE_BINS; ++bin) {
        double lambda = (double)LAMBDA_MIN + bin;
        host.cieX[bin] = (float)cie_interp(cie_x, lambda);
        host.cieY[bin] = (float)cie_interp(cie_y, lambda);
        host.cieZ[bin] = (float)cie_interp(cie_z, lambda);
        host.d65[bin] = (float)cie_interp(cie_d65, lambda);
        host.illumA[bin] = (float)illuminantASPD(lambda);
    }

    // Normalize the illuminants to unit luminance under OUR sampling
    // convention: spectrumToRGB computes Y = sum(illum * cieY) / CIE_Y_INTEGRAL,
    // so scale each illuminant to make that exactly 1. (cie1931.h's D65 is
    // already close; this removes the residual resampling error too.)
    auto normalizeLuminance = [](float *illum, const float *cieYTable) {
        double lum = 0.0;
        for (int bin = 0; bin < N_CIE_BINS; ++bin) {
            lum += (double)illum[bin] * cieYTable[bin];
        }
        float s = (float)(CIE_Y_INTEGRAL / lum);
        for (int bin = 0; bin < N_CIE_BINS; ++bin) {
            illum[bin] *= s;
        }
    };
    normalizeLuminance(host.d65, host.cieY);
    normalizeLuminance(host.illumA, host.cieY);

    host.rgb2specData = s_rgb2specData;
    host.rgb2specScale = s_rgb2specScale;
    host.rgb2specRes = res;

    cudaMemcpyToSymbol(c_spectral, &host, sizeof(SpectralTablesDev));
    checkCUDAError("initSpectralTables");

    // Verify the upload from the device's point of view before rendering.
    {
        float *dProbe = nullptr;
        float hProbe[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        cudaMalloc(&dProbe, sizeof(hProbe));
        probeSpectralTables<<<1, 1>>>(dProbe);
        cudaMemcpy(hProbe, dProbe, sizeof(hProbe), cudaMemcpyDeviceToHost);
        cudaFree(dProbe);
        checkCUDAError("probeSpectralTables");
        int cieBin538 = 538 - (int)LAMBDA_MIN;
        int d65Bin540 = 540 - (int)LAMBDA_MIN;
        if (hProbe[0] != host.cieY[cieBin538] ||
            hProbe[1] != host.d65[d65Bin540] || hProbe[2] != (float)res) {
            fprintf(stderr,
                    "FATAL: spectral tables did not reach the device "
                    "(device sees cieY=%g d65=%g res=%g scale0=%g; expected "
                    "cieY=%g d65=%g res=%d). Constant-symbol upload failed.\n",
                    hProbe[0], hProbe[1], hProbe[2], hProbe[3],
                    host.cieY[cieBin538], host.d65[d65Bin540], res);
            exit(EXIT_FAILURE);
        }
    }

    rgb2spec_free(model);
    s_loaded = true;
    printf("Spectral tables uploaded (rgb2spec res %d, %d CIE bins)\n", res,
           N_CIE_BINS);
}

void freeSpectralTables() {
    cudaFree(s_rgb2specData);  // no-op if null
    cudaFree(s_rgb2specScale); // no-op if null
    s_rgb2specData = nullptr;
    s_rgb2specScale = nullptr;
    s_loaded = false;
}

#endif // SPECTRAL
