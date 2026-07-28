#include "bxdf.h"

#include "spectral.h"
#include "volume.h" // phaseHG / sampleHGCosTheta for the phase lobe

// Index of refraction for air (the assumed exterior medium).
#define EXT_IOR 1.000277f

__host__ __device__ unsigned int materialLobeFlags(int matType) {
    switch (matType) {
    case MatType::MIRROR:
        return BXDF_SPECULAR_REFLECTION;
    case MatType::DIELECTRIC:
        // Fresnel-weighted delta reflection + delta transmission.
        return BXDF_SPECULAR_REFLECTION | BXDF_SPECULAR_TRANSMISSION;
    case MatType::MICROFACET:
        // Beckmann glossy lobe mixed with a diffuse lobe (energy split by
        // the F0 weight), both reflective.
        return BXDF_GLOSSY_REFLECTION | BXDF_DIFFUSE_REFLECTION;
    case LOBE_PHASE_HG:
        // Scatters over the full sphere; non-delta, so NEE evaluates it.
        return BXDF_DIFFUSE | BXDF_REFLECTION | BXDF_TRANSMISSION;
    case MatType::DIFFUSE:
    default:
        return BXDF_DIFFUSE_REFLECTION;
    }
}

__host__ __device__ BSDF makePhaseBSDF(float g) {
    BSDF b;
    b.lobe = LOBE_PHASE_HG;
    b.albedo = Spectrum(1.0f);
    b.specColor = Spectrum(1.0f);
    b.roughness = g; // HG asymmetry parameter
    b.intIOR = 1.0f;
    b.abbe = 0.0f;
    // No meaningful shading frame: the phase function is defined around the
    // incoming direction, handled directly in sample/eval.
    b.ns = glm::vec3(0.0f, 0.0f, 1.0f);
    b.worldToLocal = glm::mat3(1.0f);
    b.localToWorld = glm::mat3(1.0f);
    return b;
}

__host__ __device__ BSDF makeBSDF(const Material &m, const Spectrum &albedo,
                                  const Spectrum &specColor, glm::vec3 normal,
                                  const glm::vec3 &tangent,
                                  const TextureValues &texVals,
                                  const glm::vec3 &woW) {
    BSDF b;
    b.lobe = m.type;
    b.albedo = albedo;
    b.specColor = specColor;
    b.roughness = m.roughness;
    b.intIOR = m.indexOfRefraction;
    b.abbe = m.abbe;

    // Interpolated (smooth) vertex normals and back-face hits can leave the
    // shading normal pointing away from the viewer (dot(normal, woW) < 0),
    // which flips the local shading frame and produces inverted shading.
    // Reflective lobes assume the normal faces woW, so face-forward it.
    // Transmissive materials are left untouched: they need the true two-sided
    // normal to tell whether the ray is entering or exiting the surface.
    if (!hasTransmission(materialLobeFlags(m.type)) &&
        glm::dot(normal, woW) < 0.0f) {
        normal = -normal;
    }

    glm::vec3 actualNormal = normal;

    // Build a UV-aligned TBN for normal/bump mapping. Tangent-space maps
    // assume +X follows increasing U and +Y increasing V, which the BSDF's
    // arbitrary frame does not. Gram-Schmidt the tangent against the
    // (face-forwarded) normal; fall back to the BSDF frame when no UV tangent
    // exists (e.g. cubes and spheres, which pass a zero tangent).
    glm::mat3 tbn = LocalToWorld(normal);
    if (glm::dot(tangent, tangent) > 1e-12f) {
        glm::vec3 T = tangent - normal * glm::dot(normal, tangent);
        if (glm::dot(T, T) > 1e-12f) {
            T = glm::normalize(T);
            glm::vec3 B = glm::cross(normal, T);
            tbn = glm::mat3(T, B, normal);
        }
    }

    if (texVals.normal != glm::vec4(INFINITY)) {
        glm::vec3 texNormal = glm::normalize(glm::vec3(texVals.normal));
        actualNormal = glm::normalize(tbn * texNormal);
    }

    if (texVals.bump != glm::vec4(INFINITY)) {
        float du = texVals.bump.x;
        float dv = texVals.bump.y;
        // Perturb the normal along the UV-aligned tangent/bitangent.
        actualNormal = glm::normalize(actualNormal + du * tbn[0] + dv * tbn[1]);
    }

    // Freeze the shading frame around the (possibly normal/bump-mapped)
    // normal. sample, eval and pdf all use this one frame, so the pdf carried
    // into MIS weights is always consistent with the sampled direction.
    b.ns = actualNormal;
    b.worldToLocal = WorldToLocal(actualNormal);
    b.localToWorld = LocalToWorld(actualNormal);
    return b;
}

__host__ __device__ void BSDF::sample(const glm::vec3 &woW, const glm::vec2 &u,
                                      SampledWavelengths &swl,
                                      BSDFSample &s) const {
    s.weight = Spectrum(0.0f);
    s.wiW = glm::vec3(0.0f);
    s.pdf = 0.0f;
    s.eta = 1.0f;
    s.flags = BXDF_NONE;

    glm::vec3 woL = worldToLocal * woW;

    switch (lobe) {
    case MatType::DIFFUSE:
    default: {
        s.weight = sampleDiffuse(albedo, ns, u, s.wiW, s.eta);
        glm::vec3 wiL = worldToLocal * s.wiW;
        s.pdf = pdfDiffuse(woL, wiL);
        s.flags = BXDF_DIFFUSE_REFLECTION;
        break;
    }
    case MatType::MIRROR: {
        glm::mat3 w2l = worldToLocal;
        s.weight = sampleMirror(ns, w2l, woW, s.wiW, specColor, s.eta);
        s.pdf = pdfMirror(); // 0: delta lobe
        s.flags = BXDF_SPECULAR_REFLECTION;
        break;
    }
    case MatType::DIELECTRIC: {
        float ior = intIOR;
#if SPECTRAL
        if (abbe > 0.0f) {
            // Dispersive glass: one refracted direction can only be
            // correct for one wavelength, so collapse the path to its
            // hero wavelength (PBRT-style terminateSecondary) and use the
            // hero's Cauchy IOR for the Fresnel coin flip and the
            // refracted direction.
            swl.terminateSecondary();
            ior = cauchyIOR(intIOR, abbe, swl.lambda[0]);
        }
#endif
        glm::mat3 w2l = worldToLocal;
        s.weight = sampleDielectric(ns, w2l, localToWorld, woW, u.x, EXT_IOR,
                                    ior, specColor, s.wiW, s.eta);
#if SPECTRAL
        if (abbe > 0.0f) {
            // Zero the secondary channels' throughput. Not required for
            // unbiasedness (the sensor already ignores them: their pdf is
            // 0), but it keeps Russian roulette's maxComponent
            // hero-driven.
            s.weight[1] = s.weight[2] = s.weight[3] = 0.0f;
        }
#endif
        s.pdf = pdfDielectric(); // 0: delta lobe
        // Same side of the (two-sided) surface as woW -> the Fresnel coin
        // flip picked reflection; opposite side -> transmission.
        bool reflected =
            (glm::dot(s.wiW, ns) > 0.0f) == (glm::dot(woW, ns) > 0.0f);
        s.flags =
            reflected ? BXDF_SPECULAR_REFLECTION : BXDF_SPECULAR_TRANSMISSION;
        break;
    }
    case MatType::MICROFACET: {
        float tmp = (EXT_IOR - intIOR) / (EXT_IOR + intIOR);
        // Specular component based on Fresnel term (F0), diffuse
        // component scaled to conserve energy.
        float m_ks = tmp * tmp;
        Spectrum m_kd = (1.0f - m_ks) * albedo;
        // The glossy-vs-diffuse lobe choice is folded into u.x inside
        // sampleMicrofacet (it remaps u.x by m_ks), matching the mixture
        // pdf in pdfMicrofacet.
        s.weight = sampleMicrofacet(ns, worldToLocal, localToWorld, woW, m_kd,
                                    m_ks, specColor, roughness, EXT_IOR, intIOR,
                                    u, s.wiW, s.pdf, s.eta);
        s.flags =
            (u.x < m_ks) ? BXDF_GLOSSY_REFLECTION : BXDF_DIFFUSE_REFLECTION;
        break;
    }
    case LOBE_PHASE_HG: {
        // Henyey-Greenstein phase function around the incoming
        // PROPAGATION direction d = -woW. Perfect importance sampling:
        // f == pdf, so the weight is 1 (the sigma_s/sigma_t albedo is
        // applied by the medium distance sampler, not here).
        glm::vec3 d = -woW;
        float g = roughness;
        float cosT = sampleHGCosTheta(g, u.x);
        float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
        float phi = 2.0f * M_PIf * u.y;
        glm::mat3 frame = LocalToWorld(d);
        s.wiW = glm::normalize(
            frame * glm::vec3(sinT * cosf(phi), sinT * sinf(phi), cosT));
        s.pdf = phaseHG(cosT, g);
        s.weight = Spectrum(1.0f);
        s.flags = BXDF_DIFFUSE | BXDF_REFLECTION | BXDF_TRANSMISSION;
        break;
    }
    }
}

__host__ __device__ Spectrum BSDF::eval(const glm::vec3 &woW,
                                        const glm::vec3 &wiW,
                                        float &pdf) const {
    pdf = 0.0f;

    glm::vec3 woL = worldToLocal * woW;
    glm::vec3 wiL = worldToLocal * wiW;

    switch (lobe) {
    case MatType::DIFFUSE:
    default: {
        pdf = pdfDiffuse(woL, wiL);
        return evalDiffuse(albedo, woL, wiL);
    }
    case MatType::MICROFACET: {
        if (cosTheta(woL) <= 0.0f || cosTheta(wiL) <= 0.0f) {
            return Spectrum(0.0f);
        }
        float tmp = (EXT_IOR - intIOR) / (EXT_IOR + intIOR);
        float m_ks = tmp * tmp;
        Spectrum m_kd = (1.0f - m_ks) * albedo;
        glm::vec3 whL = glm::normalize(woL + wiL);
        pdf = pdfMicrofacet(m_ks, roughness, woL, wiL, whL);
        return evalMicrofacet(woL, wiL, whL, roughness, EXT_IOR, intIOR, m_kd,
                              m_ks, specColor);
    }
    case LOBE_PHASE_HG: {
        // Value == pdf (see sample); cosTheta between propagation directions.
        float p = phaseHG(glm::dot(-woW, wiW), roughness);
        pdf = p;
        return Spectrum(p);
    }
    case MatType::MIRROR:
    case MatType::DIELECTRIC:
        // Delta lobes: zero probability of evaluating an arbitrary
        // direction pair.
        return Spectrum(0.0f);
    }
}

__host__ __device__ float BSDF::pdf(const glm::vec3 &woW,
                                    const glm::vec3 &wiW) const {
    float p;
    eval(woW, wiW, p);
    return p;
}
