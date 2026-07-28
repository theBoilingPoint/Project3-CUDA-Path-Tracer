#pragma once

#include <glm/glm.hpp>
#include <thrust/random.h>

#include "bsdf.h"
#include "sceneStructs.h"

// ============================================================================
// BSDF closure interface.
//
// Every material is exposed to the integrator through one small interface:
//
//   BSDF bsdf = makeBSDF(material, albedo, specColor, normal, tangent,
//                        texVals, woW);
//   bsdf.flags()                    -> what kinds of lobes this surface has
//   bsdf.sample(woW, u, swl, out)   -> pick a direction, return weight/pdf/
//                                      flags of the sampled lobe
//   bsdf.eval(woW, wiW, pdf)        -> f(wo, wi) and the sampler's pdf for wiW
//
// The integrator never switches on the material enum: all gating (NEE
// skipping, MIS specular handling, shadow-terminator fixes, face-forwarding)
// is derived from BxDFFlags. Adding a material = adding one lobe case here +
// its flags; every integrator decision then falls out correctly.
//
// The shading frame (face-forwarding + normal/bump mapping) is built ONCE in
// makeBSDF and shared by sample/eval/pdf, so the pdf used in MIS weights is
// always consistent with the sampled direction. (Previously eval ignored
// normal/bump maps while sample applied them.)
// ============================================================================

// Lobe classification bits, PBRT-style. A material's flags() is the union of
// its lobes; a BSDFSample carries the flags of the single lobe that was
// actually sampled.
enum BxDFFlags : unsigned int {
    BXDF_NONE = 0,
    BXDF_REFLECTION = 1 << 0,
    BXDF_TRANSMISSION = 1 << 1,
    BXDF_DIFFUSE = 1 << 2,
    BXDF_GLOSSY = 1 << 3,
    BXDF_SPECULAR = 1 << 4, // delta lobe: cannot be evaluated / NEE'd

    // Common composites
    BXDF_DIFFUSE_REFLECTION = BXDF_DIFFUSE | BXDF_REFLECTION,
    BXDF_GLOSSY_REFLECTION = BXDF_GLOSSY | BXDF_REFLECTION,
    BXDF_SPECULAR_REFLECTION = BXDF_SPECULAR | BXDF_REFLECTION,
    BXDF_SPECULAR_TRANSMISSION = BXDF_SPECULAR | BXDF_TRANSMISSION,
};

__host__ __device__ inline bool hasReflection(unsigned int f) {
    return (f & BXDF_REFLECTION) != 0;
}
__host__ __device__ inline bool hasTransmission(unsigned int f) {
    return (f & BXDF_TRANSMISSION) != 0;
}
__host__ __device__ inline bool hasDiffuse(unsigned int f) {
    return (f & BXDF_DIFFUSE) != 0;
}
__host__ __device__ inline bool hasGlossy(unsigned int f) {
    return (f & BXDF_GLOSSY) != 0;
}
__host__ __device__ inline bool hasSpecular(unsigned int f) {
    return (f & BXDF_SPECULAR) != 0;
}
// Delta-only BSDF: every lobe is specular. NEE cannot evaluate it, a BSDF
// ray through it takes full MIS weight, and its reported pdf is 0.
__host__ __device__ inline bool isDeltaOnly(unsigned int f) {
    return hasSpecular(f) && !hasDiffuse(f) && !hasGlossy(f);
}

// Union of lobe flags for a material type (MatType). Usable before a BSDF is
// constructed (makeBSDF needs it for face-forwarding rules).
__host__ __device__ unsigned int materialLobeFlags(int matType);

// Extra lobe tag (outside the MatType range): the Henyey-Greenstein phase
// function of a participating medium. A medium scatter event builds a BSDF
// with this lobe (makePhaseBSDF) so it flows through exactly the same
// NEE/MIS machinery as surface closures. It scatters over the full sphere
// and is never delta, so flags-derived gating does the right thing.
constexpr int LOBE_PHASE_HG = 100;

// Result of BSDF::sample.
struct BSDFSample {
    Spectrum weight;    // f * |cos| / pdf (delta lobes fold their weight here)
    glm::vec3 wiW;      // sampled world-space incident direction
    float pdf;          // solid-angle pdf of wiW; 0 for delta lobes
    float eta;          // IOR bookkeeping for Russian roulette (1 if no
                        // interface crossing; matches the old scatterRay eta)
    unsigned int flags; // BxDFFlags of the lobe actually sampled
};

// The per-hit BSDF closure: resolved material parameters + the shading frame.
// Built once per intersection by makeBSDF; sample/eval/pdf all share it.
struct BSDF {
    int lobe; // MatType tag: which lobe set this closure dispatches to

    // Resolved color inputs (texture override + spectral uplift done by the
    // caller before construction).
    Spectrum albedo;
    Spectrum specColor;
    float roughness;
    float intIOR;
    float abbe; // dispersion (dielectric, SPECTRAL builds); 0 = off

    // Shading frame around the final shading normal (face-forwarded for
    // reflective-only materials, then normal/bump mapped).
    glm::vec3 ns;
    glm::mat3 worldToLocal;
    glm::mat3 localToWorld;

    __host__ __device__ unsigned int flags() const {
        return materialLobeFlags(lobe);
    }

    // Sample an incident direction for outgoing woW. u is a 2D uniform sample
    // (lobe selection is folded into u.x exactly as before). swl is mutable:
    // a dispersive dielectric terminates the secondary wavelengths.
    __host__ __device__ void sample(const glm::vec3 &woW, const glm::vec2 &u,
                                    SampledWavelengths &swl,
                                    BSDFSample &s) const;

    // Evaluate f(woW, wiW) and the pdf sample() would report for wiW.
    // Delta-only closures return f = 0, pdf = 0.
    __host__ __device__ Spectrum eval(const glm::vec3 &woW,
                                      const glm::vec3 &wiW, float &pdf) const;

    __host__ __device__ float pdf(const glm::vec3 &woW,
                                  const glm::vec3 &wiW) const;
};

// Build the closure for one intersection: applies the face-forwarding rule
// (skipped for transmissive materials, which need the true two-sided normal),
// the UV-aligned TBN, and normal/bump map perturbation, then freezes the
// shading frame that sample/eval/pdf share.
__host__ __device__ BSDF makeBSDF(const Material &m, const Spectrum &albedo,
                                  const Spectrum &specColor, glm::vec3 normal,
                                  const glm::vec3 &tangent,
                                  const TextureValues &texVals,
                                  const glm::vec3 &woW);

// Phase-function closure for a scatter event inside a medium with
// Henyey-Greenstein asymmetry g (stored in `roughness`; there is no shading
// frame -- the phase function is defined around the incoming direction).
__host__ __device__ BSDF makePhaseBSDF(float g);
