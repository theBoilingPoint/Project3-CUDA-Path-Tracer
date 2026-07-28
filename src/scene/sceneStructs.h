#pragma once

#include "glm/fwd.hpp"
#include "glm/glm.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <vector>

// Spectral-vs-RGB transport switch, Spectrum type and SampledWavelengths.
#include "../render/spectral.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType { SPHERE, CUBE, MESH };

enum MatType { DIFFUSE, MIRROR, DIELECTRIC, MICROFACET, TEXTURE, MEDIUM };

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 points[3];
    glm::vec3 planeNormal;
    glm::vec3 normals[3];
    glm::vec2 uvs[3];

    Triangle()
        : points{glm::vec3(), glm::vec3(), glm::vec3()},
          planeNormal(glm::vec3()),
          normals{glm::vec3(), glm::vec3(), glm::vec3()},
          uvs{glm::vec2(), glm::vec2(), glm::vec2()} {}

    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
        : points{p1, p2, p3},
          planeNormal(glm::normalize(glm::cross(p2 - p1, p3 - p2))),
          normals{planeNormal, planeNormal, planeNormal},
          uvs{glm::vec2(), glm::vec2(), glm::vec2()} {}

    Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 n1,
             glm::vec3 n2, glm::vec3 n3)
        : points{p1, p2, p3},
          planeNormal(glm::normalize(glm::cross(p2 - p1, p3 - p2))),
          normals{n1, n2, n3}, uvs{glm::vec2(), glm::vec2(), glm::vec2()} {}
};

struct BoundingBox {
    glm::vec3 min;
    glm::vec3 max;

    // Default to an empty/inverted box so accumulating via glm::min/glm::max
    // (PBRT's Union) starts from a neutral element.
    BoundingBox()
        : min(glm::vec3(std::numeric_limits<float>::max())),
          max(glm::vec3(std::numeric_limits<float>::lowest())) {}

    BoundingBox(const glm::vec3 &min, const glm::vec3 &max)
        : min(min), max(max) {}

    BoundingBox(const Triangle &triangle) {
        min = glm::min(triangle.points[0],
                       glm::min(triangle.points[1], triangle.points[2]));
        max = glm::max(triangle.points[0],
                       glm::max(triangle.points[1], triangle.points[2]));
    }

    glm::vec3 diagonal() const { return max - min; }

    float surfaceArea() const {
        glm::vec3 d = diagonal();
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    glm::vec3 offset(const glm::vec3 &p) const {
        glm::vec3 o = p - min;
        if (max.x > min.x)
            o.x /= max.x - min.x;
        if (max.y > min.y)
            o.y /= max.y - min.y;
        if (max.z > min.z)
            o.z /= max.z - min.z;
        return o;
    }

    int maxDimension() const {
        glm::vec3 d = diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
};

struct alignas(32) LinearBVHNode {
    BoundingBox bbox;
    union {
        int primitivesOffset;  // leaf
        int secondChildOffset; // interior
    };
    uint16_t nPrimitives; // 0 -> interior node
    uint8_t axis;         // interior node: xyz
};

// Host-only owner of a mesh geom's CPU-side triangle and flattened-BVH arrays.
// The Scene owns one of these per geom (parallel to `geoms`); they are uploaded
// to the device during pathtraceInit and then freed. The device-facing Geom
// below only carries the device pointers the kernel actually dereferences.
struct MeshData {
    int numTriangles = 0;
    Triangle *triangles = nullptr; // Host-side triangles (BVH-ordered)
    int numNodes = 0;
    LinearBVHNode *nodes = nullptr; // Host-side flattened BVH
};

// Indices into the scene's material and texture arrays, shared by Geom and
// ShadeableIntersection. A texture index of -1 means "no texture".
struct MaterialIDs {
    int materialId;
    int albedoTextureID;
    int normalTextureID;
    int bumpTextureID;
};

// Device-facing geometry record. Holds only what the intersection kernels
// actually read: the type tag, the mesh device pointers (dereferenced only when
// type == MESH), the transform matrices, and the material/texture indices.
// translation/rotation/scale are build-time inputs to the matrices and are kept
// local at scene-load time (see scene.cpp) rather than stored here.
struct Geom {
    enum GeomType type;

    // Mesh-only device pointers. Sphere/cube don't use these, but all geoms
    // share one array so the fields are present on every element.
    struct {
        int numTriangles;        // Used by the naive (non-BVH) traversal
        Triangle *devTriangles;  // Device-side triangles
        LinearBVHNode *devNodes; // Device-side flattened BVH
    } geometry;

    struct {
        glm::mat4 transform;
        glm::mat4 inverseTransform;
        glm::mat4 invTranspose;
    } transform;

    MaterialIDs material;
};

struct Material {
    int type;
    glm::vec3 color;
    glm::vec3 specularColor;
    float roughness;
    float emittance;
    float indexOfRefraction;
    // Dispersion (dielectrics, SPECTRAL builds): Abbe number V of the glass;
    // 0 disables dispersion (wavelength-independent IOR, original behavior).
    float abbe;
    // Emission spectrum (emitters, SPECTRAL builds): see SpectrumType.
    int spectrumType;
    float blackbodyTemp; // kelvin, used when spectrumType == SPECTRUM_BLACKBODY
    // Host-computed scale that normalizes the Planck SPD at blackbodyTemp to
    // unit luminance (set at scene load; see blackbodyLuminanceNorm).
    float blackbodyNorm;

    // --- Participating medium (TYPE "Medium") -----------------------------
    // A geom with this material encloses a scattering volume; its surface is
    // a null (invisible, non-refracting) boundary that only toggles the
    // path's inside-a-medium state. Media must be CUBE or SPHERE geoms (the
    // shadow-ray overlap test needs an analytic interval) and must not
    // overlap each other (no nested-media stack is tracked).
    glm::vec3 sigmaA;   // absorption cross-section per unit distance (RGB)
    glm::vec3 sigmaS;   // scattering cross-section per unit distance (RGB)
    float hgG;          // Henyey-Greenstein asymmetry g in (-1, 1); 0 = isotropic
    float densityScale; // global multiplier on sigmaA/sigmaS
    int heterogeneous;  // 1: procedural density in [0,1] modulates sigma
    float noiseScale;   // noise frequency, in units of the geom's local space
    int noiseOctaves;   // fbm octave count
    int mediumProfile;  // MediumProfile: shape of the procedural density field
};

// Procedural density-field shapes for heterogeneous media (JSON "PROFILE").
enum MediumProfile {
    MEDIUM_PROFILE_FBM = 0, // plain thresholded fbm filling the geom
    MEDIUM_PROFILE_CLOUD,   // cumulus: puff-cluster mass + warped erosion
    MEDIUM_PROFILE_PLUME,   // rising smoke column: cone + twist + break-up
};

/****** For Texture Loading ******/
// Device-side texture handle. Deliberately minimal: an array of these is
// uploaded to the GPU and indexed once per shading sample, so it carries only
// what the device actually uses — the hardware texture object. The backing
// cudaArray and other host-only handles needed to release the texture live
// host-side in TextureResource (pathtrace.cu) instead of bloating this struct.
struct Texture {
    cudaTextureObject_t texObj;
};

struct TextureValues {
    glm::vec4 albedo;
    glm::vec4 normal;
    glm::vec4 bump;
};

// Equirectangular (lat-long) HDR environment map. Sampled when a ray escapes
// the scene, providing both the visible background for primary rays and
// image-based lighting for bounced rays. `valid` is 0 when no map is
// configured, in which case escaped rays see a black background.
struct EnvironmentMap {
    cudaTextureObject_t texObj;
    int valid;
    float intensity;
    float rotation; // Yaw around the +Y axis, in radians (rotates the map)

    // Importance-sampling distribution (NEE + MIS). Device pointers to the
    // PBRT-style piecewise-constant 2D distribution built in Scene: one
    // conditional CDF per row over columns (h*(width+1) entries) and one
    // marginal CDF over rows (height+1 entries), both normalized to [0, 1].
    // `distValid` is 0 when no distribution is present (env sampling disabled).
    int width;
    int height;
    const float *conditionalCdf;
    const float *marginalCdf;
    int distValid;
};

// Delta (singular) lights: point and directional. These cannot be hit by BSDF
// sampling, so they are handled purely by next-event estimation (a shadow ray),
// with no MIS weight. Distinct from emissive *geometry* (area lights).
enum DeltaLightType { POINT_LIGHT, DIRECTIONAL_LIGHT };

struct DeltaLight {
    int type; // DeltaLightType
    // POINT: world position. DIRECTIONAL: unused.
    glm::vec3 position;
    // DIRECTIONAL: normalized direction the light travels (points away from the
    // source toward the scene). POINT: unused.
    glm::vec3 direction;
    // POINT: radiant intensity (W/sr); illuminance falls off as 1/dist^2.
    // DIRECTIONAL: radiance (constant, no falloff).
    glm::vec3 radiance;
    // Emission spectrum (SPECTRAL builds): see SpectrumType.
    int spectrumType;
    float blackbodyTemp;
    float blackbodyNorm; // see Material::blackbodyNorm
};
/*****************************************************************************************************************************/

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    Spectrum color; // Path throughput (product of BSDF weights along the path)
    // Accumulated radiance for this path. Emitter/env hits and NEE add into
    // this (throughput * incoming radiance * MIS weight); finalGather reads it.
    // Separated from throughput so next-event estimation can add direct-light
    // contributions mid-path without terminating.
    Spectrum radiance;
    // The wavelengths this path transports (SPECTRAL builds; empty tag struct
    // in RGB builds). Sampled once per path in generateRayFromCamera.
    SampledWavelengths swl;
    int pixelIndex;
    int remainingBounces;
    bool hasHitLight;
    // MIS bookkeeping for the ray currently being traced: the solid-angle pdf
    // of the BSDF bounce that produced it, and whether that bounce was a
    // specular/delta event (env seen through it gets full weight, no NEE).
    float bsdfPdf;
    bool specularBounce;
    float eta; // Used for Russian roulette to determine how likely this ray
               // survives
    // Index of the geom whose interior medium the ray currently travels
    // through (-1 = vacuum). Toggled when crossing a MEDIUM-material geom's
    // null boundary; medium distance sampling runs whenever this is >= 0.
    int mediumGeom;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t;
    glm::vec3 surfaceNormal;
    // World-space geometric (face) normal, used to keep secondary rays on the
    // correct side of the actual facet at grazing/silhouette angles where the
    // smooth shading normal diverges from it.
    glm::vec3 surfaceGeometricNormal;
    glm::vec3 surfaceTangent; // World-space UV tangent (zero if unavailable)
    glm::vec2 uv;
    MaterialIDs materials;
    // Index of the hit geom in the scene's geoms array (-1 on a miss). Lets the
    // shader recover the emitter's geometry (to compute its area for the
    // area-light MIS weight when a BSDF ray lands on an emissive surface).
    int hitGeomIndex;
};
