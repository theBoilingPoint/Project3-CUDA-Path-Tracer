#pragma once

#include <glm/glm.hpp>
#include <thrust/swap.h>

#include "spectral.h"
#include "warp.h"

/** Helper Functions */
/** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the angle between the normal and v */
__host__ __device__ float cosTheta(const glm::vec3 &v);
__host__ __device__ float tanTheta(const glm::vec3 &v);
__host__ __device__ void coordinateSystem(const glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3);
__host__ __device__ glm::mat3 LocalToWorld(const glm::vec3 &nor);
__host__ __device__ glm::mat3 WorldToLocal(const glm::vec3 &nor);
__host__ __device__ float fresnel(float cosThetaI, float extIOR, float intIOR);
__host__ __device__ float computeG(const glm::vec3 &v, const glm::vec3 &wh, const float roughness);
/*****************************************************************************/


/** PDFs */
__host__ __device__ float pdfDiffuse(const glm::vec3 &woL, const glm::vec3 &wiL);
__host__ __device__ float pdfMirror();
__host__ __device__ float pdfDielectric();
__host__ __device__ float pdfMicrofacet(const float m_ks, const float roughness, const glm::vec3 &woL, const glm::vec3 &wiL, const glm::vec3 &whL);
/*****************************************************************************/

/** Eval */
// Color-carrying quantities (albedo, specColour, m_kd and the returned f) are
// Spectrum: RGB in RGB builds, per-wavelength values in SPECTRAL builds.
// Directions, pdfs and Fresnel stay scalar/vec3 in both modes.
__host__ __device__ Spectrum evalDiffuse(const Spectrum &albedo, const glm::vec3 &woL, const glm::vec3 &wiL);
__host__ __device__ Spectrum evalMirror();
__host__ __device__ Spectrum evalDielectric();
__host__ __device__ Spectrum evalMicrofacet(const glm::vec3 &woL, const glm::vec3 &wiL, const glm::vec3 &whL, const float roughness, const float m_extIOR, const float m_intIOR, const Spectrum &m_kd, const float m_ks, const Spectrum &specColour);
/*****************************************************************************/

/** Bounce Directions and Return Colours */
__host__ __device__ Spectrum sampleDiffuse(const Spectrum &albedo, const glm::vec3 &normal, const glm::vec2 &sample2D, glm::vec3 &wiW, float &eta);
__host__ __device__ Spectrum sampleMirror(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::vec3 &woW, glm::vec3 &wiW, const Spectrum &specColour, float &eta);
__host__ __device__ Spectrum sampleDielectric(const glm::vec3 normal, glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const float sample1D, const float m_extIOR, const float m_intIOR, const Spectrum specColour, glm::vec3 &wiW, float &eta);
__host__ __device__ Spectrum sampleMicrofacet(const glm::vec3 &normal, const glm::mat3 &worldToLocal, const glm::mat3 &localToWorld, const glm::vec3 &woW, const Spectrum &m_kd, const float m_ks, const Spectrum &specColour, const float roughness, const float m_extIOR, const float m_intIOR, const glm::vec2 sample2D, glm::vec3 &wiW, float &pdf, float &eta);
/*****************************************************************************/