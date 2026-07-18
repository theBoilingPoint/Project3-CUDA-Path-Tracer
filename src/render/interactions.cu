#include "interactions.h"

// Index of refraction for air
#define EXT_IOR 1.000277f

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 woW,
    glm::vec3 normal, // Here normal is in world space
    glm::vec3 tangent, // World-space UV tangent (zero if unavailable)
    glm::vec3 &wiW,
    float &pdf,
    Spectrum &c,
    float &eta,
    const Material &m,
    const Spectrum &albedo,       // resolved by the caller (texture override + uplift)
    const Spectrum &specColor,    // resolved by the caller
    const TextureValues& texVals, // normal/bump maps only
    SampledWavelengths &swl,      // mutable: dispersive dielectrics terminate
                                  // secondary wavelengths (no-op in RGB builds)
    thrust::default_random_engine &rng)
{
    // Interpolated (smooth) vertex normals and back-face hits can leave the
    // shading normal pointing away from the viewer (dot(normal, woW) < 0),
    // which flips the local shading frame and produces inverted/"funny"
    // shading. Reflective lobes assume the normal faces woW, so face-forward it.
    // Dielectric is left untouched: it needs the true two-sided normal to tell
    // whether the ray is entering or exiting the surface.
    if (m.type != MatType::DIELECTRIC && glm::dot(normal, woW) < 0.0f) {
        normal = -normal;
    }

    glm::mat3 worldToLocal = WorldToLocal(normal);
    glm::mat3 localToWorld = LocalToWorld(normal);
    glm::vec3 woL = worldToLocal * woW; 
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 sample2D(u01(rng), u01(rng));

    glm::vec3 actualNormal = normal;

    // Build a UV-aligned TBN for normal/bump mapping. Tangent-space maps assume
    // +X follows increasing U and +Y increasing V, which the BSDF's arbitrary
    // frame does not. Gram-Schmidt the tangent against the (face-forwarded)
    // normal; fall back to the BSDF frame when no UV tangent exists (e.g. cubes
    // and spheres, which pass a zero tangent).
    glm::mat3 tbn = localToWorld;
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
        actualNormal =
            glm::normalize(actualNormal + du * tbn[0] + dv * tbn[1]);
    }

    // Rebuild the shading frame around the (possibly normal/bump-mapped) normal
    // so the sampled direction and the pdf reported for it share one frame --
    // otherwise the pdf carried into the MIS weight is inconsistent with the
    // sample. With no normal/bump map actualNormal == normal, so this exactly
    // reproduces the frame built above.
    worldToLocal = WorldToLocal(actualNormal);
    localToWorld = LocalToWorld(actualNormal);
    woL = worldToLocal * woW;

    // note: this is where sorting the intersections by material is going to come in very handy
    if (m.type == MatType::DIFFUSE) {
        c = sampleDiffuse(albedo, actualNormal, sample2D, wiW, eta);
        glm::vec3 wiL = worldToLocal * wiW;
        pdf = pdfDiffuse(woL, wiL);
    }
    else if (m.type == MatType::MIRROR) {
        c = sampleMirror(actualNormal, worldToLocal, woW, wiW, specColor, eta);
        pdf = pdfMirror();
    }
    else if (m.type == MatType::DIELECTRIC) {
        float intIOR = m.indexOfRefraction;
#if SPECTRAL
        if (m.abbe > 0.0f) {
            // Dispersive glass: one refracted direction can only be correct
            // for one wavelength, so collapse the path to its hero wavelength
            // (PBRT-style terminateSecondary) and use the hero's Cauchy IOR
            // for the Fresnel coin flip and the refracted direction.
            swl.terminateSecondary();
            intIOR = cauchyIOR(m.indexOfRefraction, m.abbe, swl.lambda[0]);
        }
#endif
        c = sampleDielectric(actualNormal, worldToLocal, localToWorld, woW, sample2D.x, EXT_IOR, intIOR, specColor, wiW, eta);
#if SPECTRAL
        if (m.abbe > 0.0f) {
            // Zero the secondary channels' throughput. Not required for
            // unbiasedness (the sensor already ignores them: their pdf is 0),
            // but it keeps Russian roulette's maxComponent hero-driven.
            c[1] = c[2] = c[3] = 0.0f;
        }
#endif
        pdf = pdfDielectric();
    }
    else if (m.type == MatType::MICROFACET) {
        float tmp = (EXT_IOR - m.indexOfRefraction) / (EXT_IOR + m.indexOfRefraction);
        // Specular component based on Fresnel term
        float m_ks = tmp * tmp; // This is F0
        // Diffuse component, ensuring energy conservation
        Spectrum m_kd = (1.0f - m_ks) * albedo;

        // Given that sampleMicrofacet also calculates the pdf, we can just pass it in as a parameter
        c = sampleMicrofacet(actualNormal, worldToLocal, localToWorld, woW, m_kd, m_ks, specColor, m.roughness, EXT_IOR, m.indexOfRefraction, sample2D, wiW, pdf, eta);
    }
}

__host__ __device__ void evalBSDF(glm::vec3 woW, glm::vec3 normal,
                                  glm::vec3 tangent, glm::vec3 wiW,
                                  const Material &m,
                                  const Spectrum &albedo,
                                  const Spectrum &specColor, Spectrum &f,
                                  float &pdf) {
    f = Spectrum(0.0f);
    pdf = 0.0f;

    // Match scatterRay's frame handling so the pdf returned here is consistent
    // with the one the sampler would produce (needed for a correct MIS weight).
    if (m.type != MatType::DIELECTRIC && glm::dot(normal, woW) < 0.0f) {
        normal = -normal;
    }
    glm::mat3 worldToLocal = WorldToLocal(normal);

    glm::vec3 woL = worldToLocal * woW;
    glm::vec3 wiL = worldToLocal * wiW;

    if (m.type == MatType::DIFFUSE) {
        f = evalDiffuse(albedo, woL, wiL);
        pdf = pdfDiffuse(woL, wiL);
    } else if (m.type == MatType::MICROFACET) {
        if (cosTheta(woL) > 0.0f && cosTheta(wiL) > 0.0f) {
            float tmp = (EXT_IOR - m.indexOfRefraction) /
                        (EXT_IOR + m.indexOfRefraction);
            float m_ks = tmp * tmp;
            Spectrum m_kd = (1.0f - m_ks) * albedo;
            glm::vec3 whL = glm::normalize(woL + wiL);
            f = evalMicrofacet(woL, wiL, whL, m.roughness, EXT_IOR,
                               m.indexOfRefraction, m_kd, m_ks, specColor);
            pdf = pdfMicrofacet(m_ks, m.roughness, woL, wiL, whL);
        }
    }
    // MIRROR / DIELECTRIC: delta lobes, leave f = 0, pdf = 0.
}