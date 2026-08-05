#include "pathtrace.h"

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

#include "bxdf.h"
#include "color.h"
#include "cudaUtil.h"
#include "deviceScene.h"
#include "glm/glm.hpp"
#include "intersections.h"
#include "scene.h"
#include "sceneStructs.h"
#include "spectrumData.h"
#include "utilities.h"
#include "volume.h" // Procedural atmospheric/combustion density profiles (v47).

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define ERRORCHECK 1

// Performance Improvements
#define USE_STREAM_COMPACTION 1
#define USE_MATERIAL_SORT 1
#define USE_RUSSIAN_ROULETTE 1
#define USE_BVH 1

// Visual Improvements
#define USE_ANTIALIASING 1
#define USE_CHECKERBOARD_TEXTURE 0 // This is the basic procedural texture

// Next-event estimation + multiple importance sampling. When 1, each surface
// hit also samples the lights directly (environment, area, and point/
// directional) via shadow rays, combined with BSDF sampling using the power
// heuristic -- far less noise for small/bright lights. When 0, the tracer falls
// back to pure BSDF path tracing: lights are only found by rays that happen to
// hit them, env/emitter contributions are added at full weight, and no shadow
// rays are cast. NOTE: point/directional (delta) lights can ONLY be sampled by
// NEE, so they contribute nothing when this is 0.
#define USE_MIS 1

// Firefly suppression: cap each sample's contribution so a single bounce that
// hits a tiny ultra-bright spot in the HDR (sun, softbox) or focuses a caustic
// can't dump a huge finite value into a pixel. Biases highlights slightly
// (energy loss in the brightest regions) in exchange for far less speckle.
// Tune FIREFLY_CLAMP_MAX up if highlights look dim, down if speckle remains.
#define USE_FIREFLY_CLAMP 0
#define FIREFLY_CLAMP_MAX 8.0f

// Spectral (hero-wavelength) vs RGB transport is a cross-file switch and
// therefore does NOT live here: see SPECTRAL in src/render/spectral.h.

static Scene *hst_scene = NULL;
static GuiDataContainer *guiData = NULL;

// All device allocations for the current render live here (see deviceScene.h).
// pathtraceInit fills it; the kernel launches below read pointers off of it.
static DeviceScene dev;

struct sortMaterialCondition {
    __host__ __device__ bool operator()(const ShadeableIntersection &s1,
                                        const ShadeableIntersection &s2) {
        return s1.materials.materialId < s2.materials.materialId;
    }
};

struct has_remaining_bounces {
    __host__ __device__ bool operator()(const PathSegment &path) {
        return path.remainingBounces > 0;
    }
};

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__ thrust::default_random_engine
makeSeededRandomEngine(int iter, int index, int depth, unsigned int domain) {
    // Hash independent unsigned fields instead of packing signed shifts (which
    // overflowed for 1<<31 and eventually aliased depth bits). `domain`
    // separates camera and shade streams even when their numeric depths match.
    unsigned int h =
        utilhash((unsigned int)iter ^ (domain * 0x9e3779b9u)) ^
        utilhash((unsigned int)index * 0x85ebca6bu) ^
        utilhash((unsigned int)depth * 0xc2b2ae35u);
    return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter,
                               glm::vec3 *image, float exposure, int toneMap) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix =
            displayTransform(image[index] / (float)iter, exposure, toneMap);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0f), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0f), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0f), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

void InitDataContainer(GuiDataContainer *imGuiData) { guiData = imGuiData; }

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    // All device allocation/upload lives in deviceScene.cu.
    deviceSceneInit(dev, scene);
}

void pathtraceFree() {
    // All device deallocation lives in deviceScene.cu.
    deviceSceneFree(dev);
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
                                      PathSegment *pathSegments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }

    int index = x + (y * cam.resolution.x);

    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, index, traceDepth, 0x43414d45u);
    thrust::uniform_real_distribution<float> u01(0, 1);

    PathSegment &segment = pathSegments[index];

    glm::vec3 rayOrigin = cam.position;

#if USE_ANTIALIASING
    // Jittering for anti-aliasing
    glm::vec2 offset = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f),
                                 0.5f * (u01(rng) * 2.0f - 1.0f));

    // Compute primary ray direction
    glm::vec3 rayDirection = glm::normalize(
        cam.view -
        cam.right * cam.pixelLength.x *
            ((float)x - (float)cam.resolution.x * 0.5f + offset[0]) -
        cam.up * cam.pixelLength.y *
            ((float)y - (float)cam.resolution.y * 0.5f + offset[1]));
#else
    glm::vec3 rayDirection =
        glm::normalize(cam.view -
                       cam.right * cam.pixelLength.x *
                           ((float)x - (float)cam.resolution.x * 0.5f) -
                       cam.up * cam.pixelLength.y *
                           ((float)y - (float)cam.resolution.y * 0.5f));
#endif

    // Depth of field
    float lensRadius = cam.lensRadius;
    float focalDistance = cam.focalDistance;
    if (lensRadius > 0.0f && focalDistance > 0.0f) {
        // Compute the focal point
        glm::vec3 focalPoint = rayOrigin + cam.focalDistance * rayDirection;

        // Sample point on lens (circular aperture sampling)
        glm::vec2 apartureSample = glm::vec2(u01(rng), u01(rng));
        glm::vec3 newOrigin =
            glm::vec3(lensRadius * squareToUniformDisk(apartureSample), 0.0f);

        // Offset the ray origin based on lens sampling
        rayOrigin += cam.right * newOrigin.x + cam.up * newOrigin.y;

        // Recalculate the direction to pass through the focal point
        rayDirection = glm::normalize(focalPoint - rayOrigin);
    }

    // Assign values to the path segment
    segment.ray.origin = rayOrigin;
    segment.ray.direction = rayDirection;
    segment.color = Spectrum(1.0f);    // throughput
    segment.radiance = Spectrum(0.0f); // accumulated light
#if SPECTRAL
    // Hero wavelength + 3 stratified companions for this path. Guarded so RGB
    // builds draw the exact same RNG sequence as before (bit-identical images).
    segment.swl = sampleWavelengths(u01(rng));
#endif
    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    segment.hasHitLight = false;
    // The camera ray has no preceding bounce; flag it specular so directly
    // viewed env/emitters are added at full weight (no MIS discount).
    segment.bsdfPdf = 0.0f;
    segment.specularBounce = true;
    segment.lastVertexDistance = 0.0f;
    segment.eta = 1.0f;
    // Camera starts in vacuum (a camera inside a medium is unsupported).
    segment.mediumGeom = -1;
    segment.volumeScatteringDepth = 0;
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths,
                                     PathSegment *pathSegments, Geom *geoms,
                                     int geoms_size,
                                     ShadeableIntersection *intersections) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 geometricNormal;
        glm::vec3 tangent;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec3 tmp_geoNormal;
        glm::vec3 tmp_tangent;
        glm::vec2 tmp_uv;

        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++) {
            Geom &geom = geoms[i];

            // Box/sphere don't provide a UV tangent; zero it so a stale mesh
            // tangent from a previous geom isn't reused.
            tmp_tangent = glm::vec3(0.0f);

            if (geom.type == CUBE) {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                        tmp_normal, outside);
                // Analytic primitives: shading normal is the geometric normal.
                tmp_geoNormal = tmp_normal;
            } else if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                           tmp_normal, outside);
                tmp_geoNormal = tmp_normal;
            } else if (geom.type == MESH) {
#if USE_BVH
                t = meshIntersectionTestBVH(
                    geom, pathSegment.ray, tmp_intersect, tmp_normal,
                    tmp_geoNormal, tmp_tangent, tmp_uv, outside);
#else
                t = meshIntersectionTestNaive(
                    geom, pathSegment.ray, tmp_intersect, tmp_normal,
                    tmp_geoNormal, tmp_tangent, tmp_uv, outside);
#endif
            }

            // Compute the minimum t from the intersection tests to determine
            // what scene geometry object was hit first.
            if (t > 0.0f && t_min > t) {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                geometricNormal = tmp_geoNormal;
                tangent = tmp_tangent;
                uv = tmp_uv;
            }
        }

        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
            intersections[path_index].hitGeomIndex = -1;
        } else {
            Geom hitGeom = geoms[hit_geom_index];
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materials = hitGeom.material;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].surfaceGeometricNormal = geometricNormal;
            intersections[path_index].surfaceTangent = tangent;
            intersections[path_index].uv = uv;
            intersections[path_index].hitGeomIndex = hit_geom_index;
        }
    }
}

// Sample an albedo/normal texture through the hardware texture unit. Filtering
// (bilinear) and address mode (wrap) are baked into the texture object, so this
// is just a fetch with normalized UVs.
__device__ glm::vec4 sampleTexture(Texture texture, glm::vec2 uv) {
    float4 t = tex2D<float4>(texture.texObj, uv.x, uv.y);
    return glm::vec4(t.x, t.y, t.z, t.w);
}

// Sample a bump/height map and return the UV-space height slope as (du, dv).
// One tex2Dgather fetches the height (R channel, comp 0) from all four texels
// of the 2x2 bilinear footprint around uv in a single hardware op. CUDA returns
// them counter-clockwise from the lower-left, so for texels (x0,y0)=base,
// (x1,y0)=+u, (x0,y1)=+v:
//   .w = (x0, y0)   .z = (x1, y0)   .x = (x0, y1)   .y = (x1, y1)
// Finite-differencing against the base texel gives the slope. The clamp address
// mode keeps the footprint in bounds at the edges. bumpStrength tunes the
// effect.
//
// CAVEAT: gather returns the *bilinear* footprint, which is centered on
// (uv - half a texel), so the 2x2 block can sit up to half a texel away from
// the texel that floor(uv) lands in. The local gradient is still valid for bump
// mapping, but it is not bit-identical to differencing the exact floor(uv)
// texel. If you ever need that exact alignment, offset uv by +half a texel
// (+0.5/size) before the gather.
//
// CAVEAT: tex2Dgather requires the backing cudaArray to be created with
// cudaArrayTextureGather (see initialiseTextures), and gather-enabled arrays
// have a smaller maximum dimension than regular texture fetches.
__device__ glm::vec4 sampleBump(Texture texture, glm::vec2 uv) {
    const float bumpStrength = 1.0f;

    float4 h = tex2Dgather<float4>(texture.texObj, uv.x, uv.y, 0);

    float du = (h.z - h.w) * bumpStrength; // +u neighbor minus base
    float dv = (h.x - h.w) * bumpStrength; // +v neighbor minus base

    // Stored as (du, dv) for the normal perturbation in makeBSDF.
    return glm::vec4(du, dv, 0.0f, 0.0f);
}

// Sample the equirectangular (lat-long) HDR environment map in world direction
// `dir`. Azimuth maps to U in [0, 1], elevation to V (V = 0 at the +Y pole, V =
// 1 at the -Y pole). Returns the radiance scaled by the configured intensity.
__device__ glm::vec3 sampleEnvironment(const EnvironmentMap &env,
                                       glm::vec3 dir) {
    dir = glm::normalize(dir);

    // Yaw the lookup direction around +Y by the configured rotation, which
    // spins the map horizontally about the scene.
    if (env.rotation != 0.0f) {
        float s, c;
        sincosf(env.rotation, &s, &c);
        dir = glm::vec3(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
    }

    float u = 0.5f + atan2f(dir.z, dir.x) * (0.5f * M_1_PIf);
    float v = 0.5f - asinf(glm::clamp(dir.y, -1.0f, 1.0f)) * M_1_PIf;
    float4 t = tex2D<float4>(env.texObj, u, v);
    return glm::vec3(t.x, t.y, t.z) * env.intensity;
}

// Binary search a normalized 1D CDF (n+1 entries, cdf[0]=0, cdf[n]=1). Returns
// the bin index i in [0, n-1] with cdf[i] <= u < cdf[i+1], and the fractional
// offset within that bin.
__device__ int sampleCdf1D(const float *cdf, int n, float u, float &frac) {
    int first = 0, len = n + 1;
    while (len > 0) {
        int half = len >> 1;
        int mid = first + half;
        if (cdf[mid] <= u) {
            first = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    int offset = glm::clamp(first - 1, 0, n - 1);
    float lo = cdf[offset], hi = cdf[offset + 1];
    float span = hi - lo;
    frac = span > 0.0f ? (u - lo) / span : 0.5f;
    return offset;
}

// Solid-angle pdf of the env importance distribution for a world direction
// `dir`. Mirrors sampleEnvironment's lat-long mapping (including the yaw) to
// find the (u, v) cell, reads the pdf in [0,1]^2 from the CDF differences, then
// converts to solid angle by dividing out the 2*pi^2*sin(theta) Jacobian.
__device__ float envPdf(const EnvironmentMap &env, glm::vec3 dir) {
    if (!env.distValid) {
        return 0.0f;
    }
    dir = glm::normalize(dir);
    if (env.rotation != 0.0f) {
        float s, c;
        sincosf(env.rotation, &s, &c);
        dir = glm::vec3(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
    }
    float u = 0.5f + atan2f(dir.z, dir.x) * (0.5f * M_1_PIf);
    float v = 0.5f - asinf(glm::clamp(dir.y, -1.0f, 1.0f)) * M_1_PIf;

    const int w = env.width, h = env.height;
    int iu = glm::clamp((int)(u * w), 0, w - 1);
    int iv = glm::clamp((int)(v * h), 0, h - 1);
    const float *row = env.conditionalCdf + (size_t)iv * (w + 1);
    float pdfU = (float)w * (row[iu + 1] - row[iu]);
    float pdfV = (float)h * (env.marginalCdf[iv + 1] - env.marginalCdf[iv]);

    float sinTheta = sinf(M_PIf * v);
    if (sinTheta <= 0.0f) {
        return 0.0f;
    }
    return (pdfU * pdfV) / (2.0f * M_PIf * M_PIf * sinTheta);
}

// Importance-sample a world direction toward the environment. Returns the env
// radiance in that direction; outputs the direction and its solid-angle pdf.
__device__ glm::vec3 sampleEnvDirection(const EnvironmentMap &env, float xi1,
                                        float xi2, glm::vec3 &dir,
                                        float &pdfSA) {
    const int w = env.width, h = env.height;
    float dv, du;
    int iv = sampleCdf1D(env.marginalCdf, h, xi2, dv);
    const float *row = env.conditionalCdf + (size_t)iv * (w + 1);
    int iu = sampleCdf1D(row, w, xi1, du);

    float u = ((float)iu + du) / (float)w;
    float v = ((float)iv + dv) / (float)h;
    float pdfU = (float)w * (row[iu + 1] - row[iu]);
    float pdfV = (float)h * (env.marginalCdf[iv + 1] - env.marginalCdf[iv]);

    float theta = M_PIf * v;
    float sinTheta = sinf(theta);
    if (sinTheta <= 0.0f) {
        pdfSA = 0.0f;
        dir = glm::vec3(0.0f, 1.0f, 0.0f);
        return glm::vec3(0.0f);
    }
    pdfSA = (pdfU * pdfV) / (2.0f * M_PIf * M_PIf * sinTheta);

    // (u, v) -> direction in the map frame, then un-rotate into world space
    // (inverse of the yaw sampleEnvironment applies on lookup).
    float phi = (u - 0.5f) * 2.0f * M_PIf;
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);
    glm::vec3 dirMap(sinTheta * cosPhi, cosf(theta), sinTheta * sinPhi);
    if (env.rotation != 0.0f) {
        float s, c;
        sincosf(env.rotation, &s, &c);
        dir = glm::vec3(c * dirMap.x - s * dirMap.z, dirMap.y,
                        s * dirMap.x + c * dirMap.z);
    } else {
        dir = dirMap;
    }
    return sampleEnvironment(env, dir);
}

// Power heuristic (beta = 2) MIS weight for a strategy with pdf `a` competing
// against another strategy with pdf `b` (both for the same sampled direction).
__device__ float powerHeuristic(float a, float b) {
    float a2 = a * a;
    float denom = a2 + b * b;
    return denom > 0.0f ? a2 / denom : 0.0f;
}

// Occlusion test for a shadow ray. Returns true if any scene geometry is hit at
// a distance in (epsilon, tMax). For the environment light (at infinity) tMax
// is effectively unbounded, so any hit blocks it. Medium-boundary geoms are
// null interfaces, not occluders -- they are skipped here and their
// attenuation is applied by shadowTransmittance instead.
__device__ bool anyHit(const Ray &ray, Geom *geoms, int geoms_size,
                       Material *materials, float tMax) {
    glm::vec3 tmpP, tmpN, tmpGN, tmpT;
    glm::vec2 tmpUV;
    bool outside;
    for (int i = 0; i < geoms_size; ++i) {
        Geom &geom = geoms[i];
        if (materials[geom.material.materialId].type == MatType::MEDIUM) {
            continue;
        }
        float t = -1.0f;
        if (geom.type == CUBE) {
            t = boxIntersectionTest(geom, ray, tmpP, tmpN, outside);
        } else if (geom.type == SPHERE) {
            t = sphereIntersectionTest(geom, ray, tmpP, tmpN, outside);
        } else if (geom.type == MESH) {
#if USE_BVH
            t = meshIntersectionTestBVH(geom, ray, tmpP, tmpN, tmpGN, tmpT,
                                        tmpUV, outside);
#else
            t = meshIntersectionTestNaive(geom, ray, tmpP, tmpN, tmpGN, tmpT,
                                          tmpUV, outside);
#endif
        }
        if (t > 1e-3f && t < tMax) {
            return true;
        }
    }
    return false;
}

// Host-precomputed world-space surface area. Cube and mesh values are exact;
// an affine sphere stores a close total-area proxy used only to build the
// discrete emitted-power distribution. Its conditional point PDF below uses
// the exact local-to-world area Jacobian, so NEE and hit-light MIS agree.
__device__ float lightGeomArea(const Geom &g) {
    return g.surfaceArea;
}

// Differential area scale for an affine transform M applied to a local
// sphere: dA_world = |det(M)| |M^-T n_local| dA_local.
__device__ float affineSphereAreaJacobian(const Geom &g,
                                          const glm::vec3 &nLocal) {
    const glm::vec3 ex = glm::vec3(g.transform.transform[0]);
    const glm::vec3 ey = glm::vec3(g.transform.transform[1]);
    const glm::vec3 ez = glm::vec3(g.transform.transform[2]);
    const float determinant = fabsf(glm::dot(ex, glm::cross(ey, ez)));
    const glm::vec3 inverseTransposeNormal = multiplyMV(
        g.transform.invTranspose, glm::vec4(nLocal, 0.0f));
    return determinant * glm::length(inverseTransposeNormal);
}

__device__ float lightGeomPdfAreaAtPoint(const Geom &g,
                                         const glm::vec3 &pWorld) {
    if (g.type == SPHERE) {
        const glm::vec3 pLocal = multiplyMV(
            g.transform.inverseTransform, glm::vec4(pWorld, 1.0f));
        const glm::vec3 nLocal = glm::normalize(pLocal);
        const float jacobian = affineSphereAreaJacobian(g, nLocal);
        // The canonical sphere radius is 0.5, so its local area is pi.
        return jacobian > 0.0f ? 1.0f / (M_PIf * jacobian) : 0.0f;
    }
    const float area = lightGeomArea(g);
    return area > 0.0f ? 1.0f / area : 0.0f;
}

// Sample a world-space point + outward world normal on an emitter. Cubes and
// meshes are uniform in world-area measure. Affine spheres are uniform on the
// canonical sphere and report the corresponding nonuniform world-area PDF.
// Outputs zero for unsupported or degenerate geometry.
__device__ void sampleLightGeom(const Geom &g, float u1, float u2, float u3,
                                glm::vec3 &pWorld, glm::vec3 &nWorld,
                                float &pdfArea) {
    float area = lightGeomArea(g);
    pdfArea = 0.0f;
    if (area <= 0.0f) {
        return;
    }

    // Mesh emitter: pick a triangle proportional to world area, then sample a
    // uniform barycentric point on it. Area-weighted selection makes the PDF
    // uniform over the full surface (1/totalArea), matching reverse MIS.
    if (g.type == MESH) {
        pdfArea = 1.0f / area;
        const Triangle *tris = g.geometry.devTriangles;
        int n = g.geometry.numTriangles;
        float target = u1 * area;
        int chosen = n - 1;
        float accum = 0.0f;
        for (int i = 0; i < n; ++i) {
            glm::vec3 p0 = multiplyMV(g.transform.transform,
                                      glm::vec4(tris[i].points[0], 1.0f));
            glm::vec3 p1 = multiplyMV(g.transform.transform,
                                      glm::vec4(tris[i].points[1], 1.0f));
            glm::vec3 p2 = multiplyMV(g.transform.transform,
                                      glm::vec4(tris[i].points[2], 1.0f));
            accum += 0.5f * glm::length(glm::cross(p1 - p0, p2 - p0));
            if (accum >= target) {
                chosen = i;
                break;
            }
        }
        const Triangle &tri = tris[chosen];
        float su = u2, sv = u3;
        if (su + sv > 1.0f) { // fold into the lower triangle
            su = 1.0f - su;
            sv = 1.0f - sv;
        }
        glm::vec3 lp = tri.points[0] + su * (tri.points[1] - tri.points[0]) +
                       sv * (tri.points[2] - tri.points[0]);
        pWorld = multiplyMV(g.transform.transform, glm::vec4(lp, 1.0f));
        nWorld = glm::normalize(multiplyMV(g.transform.invTranspose,
                                           glm::vec4(tri.planeNormal, 0.0f)));
        return;
    }

    glm::vec3 pLocal, nLocal;
    if (g.type == SPHERE) {
        // Uniform direction on the canonical sphere (local radius 0.5).
        float z = 1.0f - 2.0f * u1;
        float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        float phi = 2.0f * M_PIf * u2;
        nLocal = glm::vec3(r * cosf(phi), r * sinf(phi), z);
        pLocal = 0.5f * nLocal;
        const float jacobian = affineSphereAreaJacobian(g, nLocal);
        pdfArea = jacobian > 0.0f ? 1.0f / (M_PIf * jacobian) : 0.0f;
    } else { // CUBE
        pdfArea = 1.0f / area;
        glm::vec3 ex = glm::vec3(g.transform.transform[0]);
        glm::vec3 ey = glm::vec3(g.transform.transform[1]);
        glm::vec3 ez = glm::vec3(g.transform.transform[2]);
        float ax = glm::length(glm::cross(ey, ez)); // area of a +/-x face
        float ay = glm::length(glm::cross(ex, ez));
        float az = glm::length(glm::cross(ex, ey));
        float total = ax + ay + az;
        float pick = u1 * total;
        float sign = (u2 < 0.5f) ? 0.5f : -0.5f;
        float s = (u2 < 0.5f) ? (u2 * 2.0f) : ((u2 - 0.5f) * 2.0f); // in-face
        float a = s - 0.5f, b = u3 - 0.5f; // in-face coords in [-0.5, 0.5]
        if (pick < ax) {
            pLocal = glm::vec3(sign, a, b);
            nLocal = glm::vec3(sign > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
        } else if (pick < ax + ay) {
            pLocal = glm::vec3(a, sign, b);
            nLocal = glm::vec3(0.0f, sign > 0.0f ? 1.0f : -1.0f, 0.0f);
        } else {
            pLocal = glm::vec3(a, b, sign);
            nLocal = glm::vec3(0.0f, 0.0f, sign > 0.0f ? 1.0f : -1.0f);
        }
    }

    pWorld = multiplyMV(g.transform.transform, glm::vec4(pLocal, 1.0f));
    nWorld = glm::normalize(
        multiplyMV(g.transform.invTranspose, glm::vec4(nLocal, 0.0f)));
}

// Convert an area-measure pdf at a light point (pL, normal nL) into the
// solid-angle pdf as seen from shading point p. Emitters are treated as
// two-sided (|cos| at the light). Returns 0 if degenerate.
__device__ float lightPdfSolidAngle(float pdfArea, const glm::vec3 &p,
                                    const glm::vec3 &pL, const glm::vec3 &nL) {
    glm::vec3 d = pL - p;
    float dist2 = glm::dot(d, d);
    if (dist2 <= 0.0f) {
        return 0.0f;
    }
    float cosL = fabsf(glm::dot(nL, d)) / sqrtf(dist2); // |cos| at the light
    if (cosL <= 0.0f) {
        return 0.0f;
    }
    return pdfArea * dist2 / cosL;
}

__device__ int sampleAreaLightIndex(const float *cdf, int count, float u,
                                    float &selectionPmf) {
    selectionPmf = 0.0f;
    if (count <= 0) {
        return -1;
    }
    if (cdf == nullptr) {
        const int index = min((int)(u * count), count - 1);
        selectionPmf = 1.0f / count;
        return index;
    }
    const float target = fminf(fmaxf(u, 0.0f), 1.0f - 1e-7f);
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (target < cdf[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    const int index = min(lo, count - 1);
    const float lower = index > 0 ? cdf[index - 1] : 0.0f;
    selectionPmf = fmaxf(cdf[index] - lower, 0.0f);
    return selectionPmf > 0.0f ? index : -1;
}

__device__ glm::vec3 checkerboard(float u, float v, int checkerSize) {
    int u_check = static_cast<int>(floor(u * checkerSize)) % 2;
    int v_check = static_cast<int>(floor(v * checkerSize)) % 2;

    if (u_check == v_check) {
        return glm::vec3(1.0f, 1.0f, 1.0f); // white square
    } else {
        return glm::vec3(0.0f, 0.0f, 0.0f); // black square
    }
}

// ============================================================================
// RGB <-> Spectrum seam. RGB values (materials, textures, env map, lights)
// enter spectrum-land ONLY through the two uplift functions below, and leave
// it only through spectrumToRGB in finalGather. In RGB builds all three are
// identity pass-throughs, so the shading code needs no #if forks.
// ============================================================================
#if SPECTRAL

// Nearest 1nm CIE table bin for a wavelength in [LAMBDA_MIN, LAMBDA_MAX].
__device__ inline int cieBin(float lambda) {
    int bin =
        (int)(lambda + 0.5f) - (int)LAMBDA_MIN; // lambda > 0: trunc == round
    return min(max(bin, 0), N_CIE_BINS - 1);
}

// Reflectance (bounded [0,1]) RGB -> spectrum sampled at the path wavelengths,
// via the Jakob & Hanika 2019 sigmoid-polynomial model: one coefficient fetch
// per RGB value, then ~6 flops per wavelength.
__device__ Spectrum upliftReflectance(const glm::vec3 &rgb,
                                      const SampledWavelengths &swl) {
    float coeff[3];
    rs_fetchCoeffs(c_spectral.rgb2specData, c_spectral.rgb2specScale,
                   c_spectral.rgb2specRes, rgb, coeff);
    Spectrum s;
#pragma unroll
    for (int i = 0; i < NSpectrumSamples; ++i) {
        s[i] = rs_evalSigmoid(coeff, swl.lambda[i]);
    }
    return s;
}

// Named illuminant SPD at a wavelength. All illuminants are normalized to unit
// luminance, so an emitter's EMITTANCE means the same brightness for each.
// SPECTRUM_NONE and SPECTRUM_D65 both yield D65: uplifted RGB emission is
// defined relative to sRGB's whitepoint, which IS D65 (PBRT's
// RGBIlluminantSpectrum convention). Blackbody is analytic Planck scaled by
// the host-computed unit-luminance normalization (blackbodyNorm).
__device__ inline float illuminantSPD(int spectrumType, float blackbodyTemp,
                                      float blackbodyNorm, float lambda) {
    switch (spectrumType) {
    case SPECTRUM_A:
        return c_spectral.illumA[cieBin(lambda)];
    case SPECTRUM_E:
        // A constant SPD of 1 already has unit luminance under the sensor's
        // convention: spectrumToRGB divides its CIE integral by
        // CIE_Y_INTEGRAL. Applying that normalization here as well used to
        // multiply equal-energy lights by N_CIE_BINS/CIE_Y_INTEGRAL (~4.41),
        // clipping the homogeneous-absorption reference render.
        return 1.0f;
    case SPECTRUM_BLACKBODY:
        return planckSPD(lambda, blackbodyTemp) * blackbodyNorm;
    case SPECTRUM_NONE:
    case SPECTRUM_D65:
    default:
        return c_spectral.d65[cieBin(lambda)];
    }
}

// Emission (unbounded) RGB -> spectrum sampled at the path wavelengths.
// PBRT RGBIlluminantSpectrum: normalize the RGB into the sigmoid model's
// [0,1] domain, uplift, then rescale and multiply by the illuminant SPD.
// Exactly homogeneous in scale: uplift(k*rgb) == k*uplift(rgb), so scalar
// factors (EMITTANCE, 1/dist^2 falloff) pass through unchanged.
__device__ Spectrum upliftIlluminant(const glm::vec3 &rgb,
                                     const SampledWavelengths &swl,
                                     int spectrumType, float blackbodyTemp,
                                     float blackbodyNorm) {
    float m = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
    if (m <= 0.0f) {
        return Spectrum(0.0f);
    }
    float scale = 2.0f * m;
    float coeff[3];
    rs_fetchCoeffs(c_spectral.rgb2specData, c_spectral.rgb2specScale,
                   c_spectral.rgb2specRes, rgb / scale, coeff);
    Spectrum s;
#pragma unroll
    for (int i = 0; i < NSpectrumSamples; ++i) {
        s[i] = rs_evalSigmoid(coeff, swl.lambda[i]) * scale *
               illuminantSPD(spectrumType, blackbodyTemp, blackbodyNorm,
                             swl.lambda[i]);
    }
    return s;
}

// Medium extinction/scattering coefficients (unbounded, like emission) RGB ->
// spectrum at the path wavelengths. Same normalize-uplift-rescale scheme as
// upliftIlluminant, minus the illuminant SPD: sigma spectra are smooth
// reflectance-like curves scaled back to the RGB magnitude, and the scheme is
// exactly homogeneous in scale so densityScale passes through unchanged.
__device__ Spectrum upliftSigma(const glm::vec3 &rgb,
                                const SampledWavelengths &swl) {
    float m = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
    if (m <= 0.0f) {
        return Spectrum(0.0f);
    }
    float scale = 2.0f * m;
    float coeff[3];
    rs_fetchCoeffs(c_spectral.rgb2specData, c_spectral.rgb2specScale,
                   c_spectral.rgb2specRes, rgb / scale, coeff);
    Spectrum s;
#pragma unroll
    for (int i = 0; i < NSpectrumSamples; ++i) {
        s[i] = rs_evalSigmoid(coeff, swl.lambda[i]) * scale;
    }
    return s;
}

// Sensor: Monte Carlo estimate of the path's radiance spectrum -> CIE XYZ
// (via the 1931 color matching functions and the wavelength sampling pdfs)
// -> linear sRGB. Terminated secondary wavelengths contribute 0 (their pdf is
// 0); the hero pdf was pre-divided by N to compensate.
__device__ glm::vec3 spectrumToRGB(const Spectrum &L,
                                   const SampledWavelengths &swl) {
    glm::vec3 xyz(0.0f);
#pragma unroll
    for (int i = 0; i < NSpectrumSamples; ++i) {
        float p = swl.pdf[i];
        if (p <= 0.0f) {
            continue;
        }
        int bin = cieBin(swl.lambda[i]);
        float w = L[i] / p;
        xyz.x += c_spectral.cieX[bin] * w;
        xyz.y += c_spectral.cieY[bin] * w;
        xyz.z += c_spectral.cieZ[bin] * w;
    }
    xyz /= (float)NSpectrumSamples * CIE_Y_INTEGRAL;
    // XYZ -> linear sRGB (D65 whitepoint; same matrix as rgb2spec's
    // details/cie1931.h, which generated the coefficient table). Out-of-gamut
    // spectra (e.g. dispersed highlights) can go negative; the accumulation
    // buffer is signed float and display/save clamp at the end.
    return glm::vec3(3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z,
                     -0.969256f * xyz.x + 1.875991f * xyz.y + 0.041556f * xyz.z,
                     0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z);
}

#else // !SPECTRAL: identity pass-throughs

__device__ Spectrum upliftReflectance(const glm::vec3 &rgb,
                                      const SampledWavelengths &) {
    return rgb;
}

__device__ Spectrum upliftIlluminant(const glm::vec3 &rgb,
                                     const SampledWavelengths &, int, float,
                                     float) {
    return rgb;
}

__device__ Spectrum upliftSigma(const glm::vec3 &rgb,
                                const SampledWavelengths &) {
    return rgb;
}

__device__ glm::vec3 spectrumToRGB(const Spectrum &L,
                                   const SampledWavelengths &) {
    return L;
}

#endif // SPECTRAL

// Visibility of a shadow ray through participating media: 0 if an opaque
// occluder blocks the segment (media boundaries don't count), otherwise the
// product of the transmittances of every medium the segment crosses
// (analytic for homogeneous, ratio tracking for heterogeneous -- see
// volume.h).
__device__ Spectrum shadowTransmittance(const Ray &ray, float tMax,
                                        Geom *geoms, int geoms_size,
                                        Material *materials,
                                        const SampledWavelengths &swl,
                                        thrust::default_random_engine &rng,
                                        VolumeTrackingStats *volumeStats) {
    if (anyHit(ray, geoms, geoms_size, materials, tMax)) {
        return Spectrum(0.0f);
    }
    Spectrum Tr(1.0f);
    for (int i = 0; i < geoms_size; ++i) {
        const Geom &g = geoms[i];
        const Material m = materials[g.material.materialId];
        if (m.type != MatType::MEDIUM) {
            continue;
        }
        float t0, t1;
        if (!mediumInterval(g, ray, t0, t1)) {
            continue;
        }
        t0 = fmaxf(t0, 0.0f);
        t1 = fminf(t1, tMax);
        if (t1 <= t0) {
            continue;
        }
        const Spectrum smokeA =
            upliftSigma(m.sigmaA, swl) * m.densityScale;
        const Spectrum smokeS =
            upliftSigma(m.sigmaS, swl) * m.densityScale;
        if (g.volumeGrid.valid) {
            const Spectrum sootA =
                upliftSigma(m.sootSigmaA, swl) * m.densityScale;
            const Spectrum sootS =
                upliftSigma(m.sootSigmaS, swl) * m.densityScale;
            const Spectrum flameA =
                smokeA * m.flameAbsorptionScale;
            Tr *= sparseGridTransmittance(
                g, smokeA, smokeS, sootA, sootS, flameA, ray, t0, t1,
                rng, volumeStats);
        } else {
            Tr *= mediumTransmittance(g, m, smokeA + smokeS, ray, t0, t1,
                                      rng);
        }
        if (isBlack(Tr)) {
            return Spectrum(0.0f);
        }
    }
    return Tr;
}

// Unbiased stratified track-length estimator for emitted radiance along a
// camera/specular ray segment through a sparse grid. Randomized strata sample
// emission, while an independent null-collision process carries an unbiased
// ratio-tracked transmittance estimate between probes. Keeping those two point
// processes separate removes the high variance in the previous Poisson-only
// estimator (including its non-zero chance of taking no emission probe in an
// emissive brick) without repeating a prefix-transmittance walk per probe.
__device__ Spectrum sparseGridEmissionIntegral(
    const Geom &g, const Material &m, const Spectrum &smokeA,
    const Spectrum &smokeS, const Spectrum &sootA,
    const Spectrum &sootS, const Spectrum &flameA, const Ray &ray,
    float tMax, const SampledWavelengths &swl,
    thrust::default_random_engine &rng, VolumeTrackingStats *volumeStats,
    int qualityMode) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    const SparseVolumeGridDevice &grid = g.volumeGrid;
    const GridRay localRay = makeGridRay(g, ray);
    Spectrum Tr(1.0f);
    Spectrum integral(0.0f);
    float t = 0.0f;
    unsigned long long visits = 0;
    unsigned long long emptySkips = 0;
    unsigned long long nulls = 0;
    unsigned long long violations = 0;
    bool overflow = false;
    const int maxBrickSteps = grid.brickResolution.x +
                              grid.brickResolution.y +
                              grid.brickResolution.z + 12;

    for (int step = 0; step < maxBrickSteps && t < tMax; ++step) {
        const GridBrickInterval interval =
            locateGridBrick(grid, localRay, t, tMax);
        ++visits;
        if (interval.tExit <= t) {
            overflow = true;
            break;
        }
        if (interval.brickIndex < 0) {
            ++emptySkips;
            t = interval.tExit;
            continue;
        }

        const VolumeBrickMeta &meta = grid.bricks[interval.brickIndex];
        const float brickStart = t;
        const float brickEnd = interval.tExit;
        const float intervalLength = brickEnd - brickStart;
        const float extMajorant =
            gridBrickMajorant(meta, smokeA, smokeS, sootA, sootS, flameA);
        const int emissionProbeCount =
            meta.emissionPower > 0.0f
                ? (qualityMode == VOLUME_QUALITY_REFERENCE ? 4 : 1)
                : 0;
        const float probeWidth =
            emissionProbeCount > 0
                ? intervalLength / (float)emissionProbeCount
                : 0.0f;

        // Generate one extinction candidate and retain it across emission
        // strata. Discarding an exponential overshoot at every probe would
        // change the null-collision process and bias transmittance.
        float nextTrackingT = FLT_MAX;
        if (extMajorant > 0.0f) {
            nextTrackingT =
                brickStart -
                logf(fmaxf(1.0f - u01(rng), 1e-7f)) / extMajorant;
        }
        int candidateCount = 0;
        for (int probe = 0; probe <= emissionProbeCount; ++probe) {
            const float probeT =
                probe < emissionProbeCount
                    ? brickStart +
                          ((float)probe + u01(rng)) * probeWidth
                    : brickEnd;

            // Advance the independent ratio-tracking process to this probe.
            while (nextTrackingT < probeT &&
                   candidateCount < 1000000) {
                ++candidateCount;
                CombustionFieldSample trackingFields;
                const glm::vec3 trackingLocal =
                    localRay.origin +
                    nextTrackingT * localRay.direction;
                Spectrum sigTx(0.0f);
                if (sampleCombustionGridLocal(
                        grid, trackingLocal, trackingFields)) {
                    sigTx =
                        gridSigmaA(trackingFields, smokeA, sootA, flameA) +
                        gridSigmaS(trackingFields, smokeS, sootS);
                }
                if (maxComponent(sigTx) >
                    extMajorant * (1.0f + 2e-4f) + 1e-6f) {
                    ++violations;
                }
                Tr *= (Spectrum(extMajorant) - sigTx) /
                      extMajorant;
                ++nulls;

                const float maxTr = maxComponent(Tr);
                if (maxTr < 0.025f) {
                    constexpr float q = 0.75f;
                    if (u01(rng) < q) {
                        recordVolumeTrackingStats(
                            volumeStats, visits, emptySkips, nulls, 0,
                            violations, 0);
                        return integral;
                    }
                    Tr /= 1.0f - q;
                }
                nextTrackingT -=
                    logf(fmaxf(1.0f - u01(rng), 1e-7f)) /
                    extMajorant;
            }
            if (candidateCount >= 1000000) {
                overflow = true;
                break;
            }
            if (probe >= emissionProbeCount) {
                continue;
            }

            CombustionFieldSample emissionFields;
            const glm::vec3 emissionLocal =
                localRay.origin + probeT * localRay.direction;
            if (!sampleCombustionGridLocal(
                    grid, emissionLocal, emissionFields)) {
                continue;
            }
            const float source =
                gridEmissionSource(emissionFields, m.sootEmission);
            if (source > 0.0f) {
                const float temperature = glm::clamp(
                    emissionFields.temperature * m.temperatureScale,
                    500.0f, 12000.0f);
                const Spectrum Le = upliftIlluminant(
                    m.color * m.emittance, swl, m.spectrumType,
                    temperature, m.blackbodyNorm);
                integral += Tr * Le * (source * probeWidth);
            }
        }
        if (overflow) {
            break;
        }
        t = brickEnd;
    }
    if (t < tMax) {
        overflow = true;
    }
    recordVolumeTrackingStats(volumeStats, visits, emptySkips, nulls, 0,
                              violations, overflow ? 1 : 0);
    return overflow ? Spectrum(0.0f) : integral;
}

#if USE_MIS
// World-ray interval through the sparse grid's local AABB. Ray directions are
// normalized in world space, and the inverse transform deliberately leaves the
// local direction unnormalized, so t remains a world-space distance.
__device__ bool sparseGridInterval(const Geom &g, const glm::vec3 &origin,
                                   const glm::vec3 &direction, float &t0,
                                   float &t1) {
    const SparseVolumeGridDevice &grid = g.volumeGrid;
    const glm::vec3 ro = glm::vec3(
        g.transform.inverseTransform * glm::vec4(origin, 1.0f));
    const glm::vec3 rd = glm::vec3(
        g.transform.inverseTransform * glm::vec4(direction, 0.0f));
    t0 = -FLT_MAX;
    t1 = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(rd[axis]) < 1e-9f) {
            if (ro[axis] < grid.localBoundsMin[axis] ||
                ro[axis] > grid.localBoundsMax[axis]) {
                return false;
            }
            continue;
        }
        const float invD = 1.0f / rd[axis];
        const float ta = (grid.localBoundsMin[axis] - ro[axis]) * invD;
        const float tb = (grid.localBoundsMax[axis] - ro[axis]) * invD;
        t0 = fmaxf(t0, fminf(ta, tb));
        t1 = fminf(t1, fmaxf(ta, tb));
    }
    return t1 > fmaxf(t0, 0.0f);
}

// Uniform solid angle followed by uniform distance through the grid interval.
// With dV = r^2 dr dOmega its world-volume density is
//   1 / (4 pi * intervalLength * r^2),
// which cancels the singular 1/r^2 geometry term of volume-emission NEE.
// This is only mixed in when the shading point is inside the selected grid;
// there every direction intersects the convex grid AABB and no samples are
// wasted on directions that miss the emitter.
__device__ bool sampleSparseVolumeInverseSquare(
    const Geom &g, const glm::vec3 &p, const glm::vec2 &uDirection,
    float uDistance, float maxDistance, glm::vec3 &pLight,
    CombustionFieldSample &fields, float &pdfWorldVolume) {
    const float z = 1.0f - 2.0f * uDirection.x;
    const float radial = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * M_PIf * uDirection.y;
    const glm::vec3 direction(radial * cosf(phi),
                              radial * sinf(phi), z);
    float t0, t1;
    if (!sparseGridInterval(g, p, direction, t0, t1)) {
        return false;
    }
    const float begin = fmaxf(t0, 0.0f);
    const float end = fminf(t1, begin + maxDistance);
    const float intervalLength = end - begin;
    if (intervalLength <= 0.0f) {
        return false;
    }
    const float distance = begin + uDistance * intervalLength;
    if (distance <= 1e-8f) {
        return false;
    }
    pLight = p + distance * direction;
    const glm::vec3 pLocal = glm::vec3(
        g.transform.inverseTransform * glm::vec4(pLight, 1.0f));
    if (!sampleCombustionGridLocal(g.volumeGrid, pLocal, fields)) {
        fields = {};
    }
    pdfWorldVolume =
        1.0f /
        (4.0f * M_PIf * intervalLength * distance * distance);
    return pdfWorldVolume > 0.0f && isfinite(pdfWorldVolume);
}

__device__ float sparseVolumeInverseSquarePdf(
    const Geom &g, const glm::vec3 &p, const glm::vec3 &pLight,
    float maxDistance) {
    const glm::vec3 d = pLight - p;
    const float dist2 = glm::dot(d, d);
    if (dist2 <= 1e-16f) {
        return 0.0f;
    }
    const float distance = sqrtf(dist2);
    const glm::vec3 direction = d / distance;
    float t0, t1;
    if (!sparseGridInterval(g, p, direction, t0, t1)) {
        return 0.0f;
    }
    const float begin = fmaxf(t0, 0.0f);
    const float end = fminf(t1, begin + maxDistance);
    const float intervalLength = end - begin;
    if (intervalLength <= 0.0f ||
        distance < begin - 2e-4f ||
        distance > end + 2e-4f) {
        return 0.0f;
    }
    return 1.0f /
           (4.0f * M_PIf * intervalLength * dist2);
}

// Sample one point from the two-level emitted-power hierarchy of all sparse
// combustion grids. This is the only volume-emission strategy at diffuse
// surface and real medium vertices; segment emission is reserved for primary
// and specular rays, so the two techniques are mutually exclusive and cannot
// double count.
__device__ void sampleSparseVolumeDirect(
    PathSegment &pathSegment, const BSDF &bsdf, const glm::vec3 &p,
    bool isSurface, const glm::vec3 &woW, const glm::vec3 &Ng,
    Geom *geoms, int geoms_size, Material *materials,
    thrust::default_random_engine &rng, VolumeTrackingStats *volumeStats,
    const VolumeIntegratorSettings &volumeSettings) {
    if (volumeSettings.debugMode == VOLUME_DEBUG_INDIRECT_VOLUME) {
        return;
    }
    thrust::uniform_real_distribution<float> u01(0, 1);
    float totalPower = 0.0f;
    for (int i = 0; i < geoms_size; ++i) {
        const Geom &candidate = geoms[i];
        if (!candidate.volumeGrid.valid) {
            continue;
        }
        const Material &cm = materials[candidate.material.materialId];
        totalPower += candidate.volumeGrid.totalEmissionPower *
                      candidate.volumeGrid.localToWorldVolume *
                      cm.emittance;
    }
    if (totalPower <= 0.0f) {
        return;
    }

    const float select = u01(rng) * totalPower;
    float cumulative = 0.0f;
    int selectedIndex = -1;
    float selectedPower = 0.0f;
    for (int i = 0; i < geoms_size; ++i) {
        const Geom &candidate = geoms[i];
        if (!candidate.volumeGrid.valid) {
            continue;
        }
        const Material &cm = materials[candidate.material.materialId];
        const float power = candidate.volumeGrid.totalEmissionPower *
                            candidate.volumeGrid.localToWorldVolume *
                            cm.emittance;
        cumulative += power;
        if (select <= cumulative && power > 0.0f) {
            selectedIndex = i;
            selectedPower = power;
            break;
        }
    }
    if (selectedIndex < 0 || selectedPower <= 0.0f) {
        return;
    }

    const Geom &lightGeom = geoms[selectedIndex];
    const Material &lightMaterial =
        materials[lightGeom.material.materialId];
    const glm::vec3 pLocal = glm::vec3(
        lightGeom.transform.inverseTransform * glm::vec4(p, 1.0f));
    const bool insideGrid =
        glm::all(glm::greaterThanEqual(
            pLocal, lightGeom.volumeGrid.localBoundsMin)) &&
        glm::all(glm::lessThanEqual(
            pLocal, lightGeom.volumeGrid.localBoundsMax));
    const glm::vec3 localBrickExtent =
        (lightGeom.volumeGrid.localBoundsMax -
         lightGeom.volumeGrid.localBoundsMin) /
        glm::vec3(lightGeom.volumeGrid.brickResolution);
    const glm::mat3 localToWorld(lightGeom.transform.transform);
    const float nearFieldRadius =
        1.5f *
        fmaxf(glm::length(localToWorld *
                          glm::vec3(localBrickExtent.x, 0.0f, 0.0f)),
              fmaxf(glm::length(localToWorld *
                                glm::vec3(0.0f, localBrickExtent.y, 0.0f)),
                    glm::length(localToWorld *
                                glm::vec3(0.0f, 0.0f,
                                          localBrickExtent.z))));
    // The hierarchy remains the workhorse and targets hot cells. A 10% local
    // inverse-square component is enough to bound the r->0 weight while
    // avoiding the high null-sample rate of a whole-grid 50/50 mixture.
    const float hierarchyProbability = insideGrid ? 0.9f : 1.0f;

    glm::vec3 pLight(0.0f);
    CombustionFieldSample lightFields{};
    float hierarchyPdfWorld = 0.0f;
    float inverseSquarePdfWorld = 0.0f;
    if (u01(rng) < hierarchyProbability) {
        const GridEmissionPoint ep = sampleGridEmissionPoint(
            lightGeom.volumeGrid, u01(rng), u01(rng),
            glm::vec3(u01(rng), u01(rng), u01(rng)));
        if (!ep.valid) {
            return;
        }
        pLight = glm::vec3(lightGeom.transform.transform *
                           glm::vec4(ep.localPosition, 1.0f));
        lightFields = ep.fields;
        hierarchyPdfWorld =
            ep.pdfLocalVolume /
            lightGeom.volumeGrid.localToWorldVolume;
        if (insideGrid) {
            inverseSquarePdfWorld =
                sparseVolumeInverseSquarePdf(
                    lightGeom, p, pLight, nearFieldRadius);
        }
    } else {
        if (!sampleSparseVolumeInverseSquare(
                lightGeom, p, glm::vec2(u01(rng), u01(rng)), u01(rng),
                nearFieldRadius, pLight, lightFields,
                inverseSquarePdfWorld)) {
            return;
        }
        const glm::vec3 sampledLocal = glm::vec3(
            lightGeom.transform.inverseTransform *
            glm::vec4(pLight, 1.0f));
        hierarchyPdfWorld =
            gridEmissionPointPdfLocal(lightGeom.volumeGrid, sampledLocal) /
            lightGeom.volumeGrid.localToWorldVolume;
    }
    const glm::vec3 d = pLight - p;
    const float dist2 = glm::dot(d, d);
    if (dist2 <= 1e-8f) {
        return;
    }
    const float dist = sqrtf(dist2);
    const glm::vec3 wi = d / dist;
    if (isSurface && glm::dot(wi, Ng) <= 0.0f) {
        return;
    }
    float closurePdf;
    const Spectrum f = bsdf.eval(woW, wi, closurePdf);
    const float cosFactor = isSurface ? glm::dot(wi, bsdf.ns) : 1.0f;
    if (cosFactor <= 0.0f || isBlack(f)) {
        return;
    }

    Ray shadowRay;
    shadowRay.origin = isSurface ? p + Ng * 1e-3f : p;
    const glm::vec3 shadowD = pLight - shadowRay.origin;
    const float shadowDistance = glm::length(shadowD);
    shadowRay.direction = shadowD / shadowDistance;
    const Spectrum Tr = shadowTransmittance(
        shadowRay, shadowDistance * (1.0f - 2e-4f), geoms, geoms_size,
        materials, pathSegment.swl, rng, volumeStats);
    if (isBlack(Tr)) {
        return;
    }

    const float source =
        gridEmissionSource(lightFields, lightMaterial.sootEmission);
    if (source <= 0.0f) {
        return;
    }
    const float temperature = glm::clamp(
        lightFields.temperature * lightMaterial.temperatureScale, 500.0f,
        12000.0f);
    const Spectrum Le = upliftIlluminant(
        lightMaterial.color * lightMaterial.emittance, pathSegment.swl,
        lightMaterial.spectrumType, temperature,
        lightMaterial.blackbodyNorm);
    const float selectPdf = selectedPower / totalPower;
    const float pdfWorldVolume =
        selectPdf *
        (hierarchyProbability * hierarchyPdfWorld +
         (1.0f - hierarchyProbability) * inverseSquarePdfWorld);
    if (pdfWorldVolume <= 0.0f) {
        return;
    }

    pathSegment.radiance += pathSegment.color * f * cosFactor * Tr * Le *
                            (source / (dist2 * pdfWorldVolume));
}

// Next-event estimation toward every light type (environment, area, delta),
// shared by surface hits and medium scatter events: a medium event passes the
// phase-function closure (makePhaseBSDF) and isSurface = false, which drops
// the surface-only parts (geometric-normal gates, the cosine factor, and the
// normal-offset shadow origin). Delta-only closures must not call this (their
// f cannot be evaluated for a given direction).
__device__ void sampleDirectLighting(
    PathSegment &pathSegment, const BSDF &bsdf, const glm::vec3 &p,
    bool isSurface, const glm::vec3 &woW, const glm::vec3 &Ng,
    const EnvironmentMap &envMap, Geom *geoms, int geoms_size,
    Material *materials, Geom *lights, int numLights,
    const float *areaLightCdf, DeltaLight *deltaLights, int numDeltaLights,
    thrust::default_random_engine &rng,
    VolumeTrackingStats *volumeStats,
    const VolumeIntegratorSettings &volumeSettings) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 shadowOrigin = isSurface ? p + Ng * 1e-3f : p;

    if (volumeSettings.debugMode != VOLUME_DEBUG_SURFACE_FIRE ||
        isSurface) {
        sampleSparseVolumeDirect(pathSegment, bsdf, p, isSurface, woW, Ng,
                                 geoms, geoms_size, materials, rng,
                                 volumeStats, volumeSettings);
    }
    if (volumeSettings.debugMode == VOLUME_DEBUG_DIRECT_VOLUME ||
        volumeSettings.debugMode == VOLUME_DEBUG_INDIRECT_VOLUME ||
        volumeSettings.debugMode == VOLUME_DEBUG_SURFACE_FIRE) {
        return;
    }

    // --- Environment light ---
    // Sample a direction from the env's luminance distribution, evaluate the
    // closure for it, and add its (transmittance-weighted) contribution
    // MIS-weighted against closure sampling.
    if (envMap.valid && envMap.distValid) {
        glm::vec3 lightDir;
        float lightPdf;
        glm::vec3 Le =
            sampleEnvDirection(envMap, u01(rng), u01(rng), lightDir, lightPdf);
        if (lightPdf > 0.0f &&
            (!isSurface || glm::dot(lightDir, Ng) > 0.0f)) {
            float bsdfPdfL;
            Spectrum f = bsdf.eval(woW, lightDir, bsdfPdfL);
            float cosFactor = isSurface ? glm::dot(lightDir, bsdf.ns) : 1.0f;
            if (bsdfPdfL > 0.0f && cosFactor > 0.0f && !isBlack(f)) {
                Ray shadowRay;
                shadowRay.origin = shadowOrigin;
                shadowRay.direction = lightDir;
                Spectrum Tr =
                    shadowTransmittance(shadowRay, FLT_MAX, geoms, geoms_size,
                                        materials, pathSegment.swl, rng,
                                        volumeStats);
                if (!isBlack(Tr)) {
                    float weight = powerHeuristic(lightPdf, bsdfPdfL);
                    Spectrum LeS = upliftIlluminant(
                        Le, pathSegment.swl, SPECTRUM_NONE, 0.0f, 0.0f);
                    pathSegment.radiance += pathSegment.color * f * cosFactor *
                                            Tr * LeS * weight / lightPdf;
                }
            }
        }
    }

    // --- Area lights ---
    // Pick one emitter from the host-built emitted-power CDF, then sample a
    // point on it. The selected PMF is part of the solid-angle PDF and is also
    // stored on the source geom for the reverse/hit-light MIS calculation.
    if (numLights > 0) {
        float selectionPmf;
        int li = sampleAreaLightIndex(areaLightCdf, numLights, u01(rng),
                                      selectionPmf);
        if (li < 0 || selectionPmf <= 0.0f) {
            // Keep sampling delta lights below even if an unsupported area
            // emitter somehow entered the distribution.
            li = 0;
            selectionPmf = 0.0f;
        }
        Geom L = lights[li];
        glm::vec3 pL, nL;
        float pdfArea;
        sampleLightGeom(L, u01(rng), u01(rng), u01(rng), pL, nL, pdfArea);
        if (pdfArea > 0.0f) {
            glm::vec3 d = pL - p;
            float dist = sqrtf(glm::dot(d, d));
            glm::vec3 lightDir = d / dist;
            float pdfSA = lightPdfSolidAngle(pdfArea, p, pL, nL) *
                          selectionPmf;
            float cosFactor = isSurface ? glm::dot(lightDir, bsdf.ns) : 1.0f;
            if (pdfSA > 0.0f && cosFactor > 0.0f &&
                (!isSurface || glm::dot(lightDir, Ng) > 0.0f)) {
                float bsdfPdfL;
                Spectrum f = bsdf.eval(woW, lightDir, bsdfPdfL);
                if (!isBlack(f)) {
                    // Re-derive the ray from the OFFSET origin and stop just
                    // short of the light with a RELATIVE epsilon. Using the
                    // un-offset distance here is wrong: when the normal points
                    // at the light, the origin offset brings the light's own
                    // surface inside tMax and the light "shadows" its own
                    // sample -- a dark cap on any surface directly facing an
                    // area light.
                    Ray shadowRay;
                    shadowRay.origin = shadowOrigin;
                    glm::vec3 sd = pL - shadowRay.origin;
                    float sdist = glm::length(sd);
                    shadowRay.direction = sd / sdist;
                    Spectrum Tr = shadowTransmittance(
                        shadowRay, sdist * (1.0f - 1e-3f), geoms, geoms_size,
                        materials, pathSegment.swl, rng, volumeStats);
                    if (!isBlack(Tr)) {
                        Material lMat = materials[L.material.materialId];
                        Spectrum Le = upliftIlluminant(
                            lMat.color * lMat.emittance, pathSegment.swl,
                            lMat.spectrumType, lMat.blackbodyTemp,
                            lMat.blackbodyNorm);
                        float weight = powerHeuristic(pdfSA, bsdfPdfL);
                        pathSegment.radiance += pathSegment.color * f *
                                                cosFactor * Tr * Le * weight /
                                                pdfSA;
                    }
                }
            }
        }
    }

    // --- Delta (point/directional) lights ---
    // Delta lights can't be hit by closure sampling, so each is a single
    // shadow-ray sample with no MIS weight (weight = 1).
    for (int li = 0; li < numDeltaLights; ++li) {
        DeltaLight dl = deltaLights[li];
        glm::vec3 lightDir;
        float dist;
        glm::vec3 Li;
        if (dl.type == POINT_LIGHT) {
            glm::vec3 d = dl.position - p;
            float dist2 = glm::dot(d, d);
            dist = sqrtf(dist2);
            lightDir = d / dist;
            Li = dl.radiance / dist2; // inverse-square falloff
        } else {                      // DIRECTIONAL_LIGHT
            lightDir = -glm::normalize(dl.direction);
            dist = FLT_MAX;
            Li = dl.radiance;
        }
        float cosFactor = isSurface ? glm::dot(lightDir, bsdf.ns) : 1.0f;
        if (cosFactor <= 0.0f ||
            (isSurface && glm::dot(lightDir, Ng) <= 0.0f)) {
            continue;
        }
        float bsdfPdfL;
        Spectrum f = bsdf.eval(woW, lightDir, bsdfPdfL);
        if (isBlack(f)) {
            continue;
        }
        Ray shadowRay;
        shadowRay.origin = shadowOrigin;
        shadowRay.direction = lightDir;
        float tMax = (dl.type == POINT_LIGHT) ? dist - 1e-3f : FLT_MAX;
        Spectrum Tr = shadowTransmittance(shadowRay, tMax, geoms, geoms_size,
                                          materials, pathSegment.swl, rng,
                                          volumeStats);
        if (!isBlack(Tr)) {
            Spectrum LiS = upliftIlluminant(Li, pathSegment.swl,
                                            dl.spectrumType, dl.blackbodyTemp,
                                            dl.blackbodyNorm);
            pathSegment.radiance +=
                pathSegment.color * f * cosFactor * Tr * LiS;
        }
    }
}
#endif // USE_MIS

__device__ glm::vec3 volumeDebugRamp(float t) {
    t = glm::clamp(t, 0.0f, 1.0f);
    if (t < 0.35f) {
        return glm::mix(glm::vec3(0.015f, 0.0f, 0.02f),
                        glm::vec3(0.65f, 0.015f, 0.0f), t / 0.35f);
    }
    if (t < 0.72f) {
        return glm::mix(glm::vec3(0.65f, 0.015f, 0.0f),
                        glm::vec3(1.0f, 0.55f, 0.02f),
                        (t - 0.35f) / 0.37f);
    }
    return glm::mix(glm::vec3(1.0f, 0.55f, 0.02f),
                    glm::vec3(1.0f), (t - 0.72f) / 0.28f);
}

// Deterministic fixed-step compositing is intentionally confined to field
// diagnostics. Reference radiance uses the null-collision estimators above.
__device__ glm::vec3 renderSparseVolumeDiagnostic(
    const Geom &g, const Material &m, const Ray &ray, float tMax,
    int debugMode) {
    constexpr int steps = 128;
    const float dt = tMax / (float)steps;
    glm::vec3 color(0.0f);
    float transmittance = 1.0f;
    const GridRay localRay = makeGridRay(g, ray);

    for (int i = 0; i < steps && transmittance > 0.005f; ++i) {
        const float t = ((float)i + 0.5f) * dt;
        CombustionFieldSample f;
        if (!sampleCombustionGridLocal(
                g.volumeGrid, localRay.origin + t * localRay.direction, f)) {
            continue;
        }

        const glm::vec3 sigA =
            m.densityScale *
            (m.sigmaA * f.density + m.sootSigmaA * f.soot +
             m.sigmaA * (m.flameAbsorptionScale * f.reaction));
        const glm::vec3 sigS =
            m.densityScale *
            (m.sigmaS * f.density + m.sootSigmaS * f.soot);
        glm::vec3 sampleColor(0.0f);
        float strength = 0.0f;
        if (debugMode == VOLUME_DEBUG_TEMPERATURE) {
            strength =
                glm::clamp((f.temperature - 293.15f) / 2500.0f, 0.0f, 1.0f);
            sampleColor = volumeDebugRamp(strength);
        } else if (debugMode == VOLUME_DEBUG_DENSITY) {
            strength = glm::clamp(f.density, 0.0f, 1.0f);
            sampleColor = glm::vec3(strength);
        } else if (debugMode == VOLUME_DEBUG_FUEL) {
            strength = glm::clamp(f.fuel, 0.0f, 1.0f);
            sampleColor = strength * glm::vec3(0.18f, 0.85f, 0.08f);
        } else if (debugMode == VOLUME_DEBUG_SOOT) {
            strength = glm::clamp(f.soot, 0.0f, 1.0f);
            sampleColor = strength * glm::vec3(0.72f, 0.62f, 0.52f);
        } else if (debugMode == VOLUME_DEBUG_REACTION) {
            strength = glm::clamp(f.reaction, 0.0f, 1.0f);
            sampleColor =
                strength * glm::vec3(1.0f, 0.20f + 0.65f * strength, 0.01f);
        } else if (debugMode == VOLUME_DEBUG_EMISSION) {
            const float thermal =
                powf(glm::clamp(f.temperature / 2600.0f, 0.0f, 2.0f), 4.0f);
            strength = glm::clamp(
                gridEmissionSource(f, m.sootEmission) * thermal, 0.0f, 1.0f);
            sampleColor = volumeDebugRamp(
                glm::clamp((f.temperature - 600.0f) / 2400.0f, 0.0f, 1.0f)) *
                          strength;
        } else if (debugMode == VOLUME_DEBUG_SIGMA_A) {
            strength = glm::clamp(maxComponent(sigA), 0.0f, 1.0f);
            sampleColor = glm::clamp(sigA, glm::vec3(0.0f), glm::vec3(1.0f));
        } else if (debugMode == VOLUME_DEBUG_SIGMA_S) {
            strength = glm::clamp(maxComponent(sigS), 0.0f, 1.0f);
            sampleColor = glm::clamp(sigS, glm::vec3(0.0f), glm::vec3(1.0f));
        } else if (debugMode == VOLUME_DEBUG_SIGMA_T) {
            strength =
                glm::clamp(maxComponent(sigA + sigS), 0.0f, 1.0f);
            sampleColor =
                glm::clamp(sigA + sigS, glm::vec3(0.0f), glm::vec3(1.0f));
        } else if (debugMode == VOLUME_DEBUG_VELOCITY) {
            const float speed = glm::length(f.velocity);
            strength = glm::clamp(speed * 3.0f, 0.0f, 1.0f);
            const glm::vec3 direction =
                speed > 1e-6f ? f.velocity / speed : glm::vec3(0.0f);
            sampleColor = (0.5f + 0.5f * direction) * strength;
        } else if (debugMode == VOLUME_DEBUG_MAJORANT) {
            const GridBrickInterval interval =
                locateGridBrick(g.volumeGrid, localRay, t, tMax);
            float majorant = 0.0f;
            if (interval.brickIndex >= 0) {
                const VolumeBrickMeta &meta =
                    g.volumeGrid.bricks[interval.brickIndex];
                majorant =
                    m.densityScale *
                    (meta.maxDensity *
                         maxComponent(m.sigmaA + m.sigmaS) +
                     meta.maxSoot *
                         maxComponent(m.sootSigmaA + m.sootSigmaS) +
                     meta.maxReaction *
                         maxComponent(m.sigmaA *
                                      m.flameAbsorptionScale));
            }
            strength = 1.0f - expf(-0.35f * majorant);
            sampleColor =
                glm::vec3(strength, 0.15f * strength, 1.0f - strength);
        }

        const float alpha =
            1.0f - expf(-fmaxf(strength, 0.0f) * dt * 6.0f);
        color += transmittance * sampleColor * alpha;
        transmittance *= 1.0f - alpha;
    }
    return color;
}

__global__ void shade(int iter, int depth, int num_paths,
                      ShadeableIntersection *shadeableIntersections,
                      PathSegment *pathSegments, Material *materials,
                      Texture *albedoTextures, Texture *normalTextures,
                      Texture *bumpTextures, EnvironmentMap envMap, Geom *geoms,
                      int geoms_size, Geom *lights, int numLights,
                      const float *areaLightCdf, DeltaLight *deltaLights,
                      int numDeltaLights,
                      VolumeTrackingStats *volumeStats,
                      VolumeIntegratorSettings volumeSettings) {
    // As long as we enter here, it means the ray has remaining bounces > 0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

#if !USE_STREAM_COMPACTION
    if (pathSegments[idx].remainingBounces <= 0) {
        return;
    }
#endif

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment &pathSegment = pathSegments[idx];
    if (intersection.t <= 0.0f) {
        // Ray escaped the scene: add the environment radiance along it. This is
        // both the visible background (primary rays) and image-based lighting
        // (bounced rays). Because we ALSO sample the env directly via NEE at
        // the previous surface, this BSDF-sampled contribution must be
        // MIS-weighted to avoid double counting. Specular bounces and the
        // primary ray (both flagged specularBounce) have no competing NEE, so
        // they take full weight.
        if (envMap.valid) {
            glm::vec3 dir = pathSegment.ray.direction;
            Spectrum Le =
                upliftIlluminant(sampleEnvironment(envMap, dir),
                                 pathSegment.swl, SPECTRUM_NONE, 0.0f, 0.0f);
            float weight = 1.0f;
#if USE_MIS
            if (!pathSegment.specularBounce && envMap.distValid) {
                float lightPdf = envPdf(envMap, dir);
                weight = powerHeuristic(pathSegment.bsdfPdf, lightPdf);
            }
#endif
            pathSegment.radiance += pathSegment.color * Le * weight;
            pathSegment.hasHitLight = true;
        }
        pathSegment.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(
        iter, pathSegment.pixelIndex, depth, 0x53484144u);
    thrust::uniform_real_distribution<float> u01(0, 1);

    // --- Participating medium: march the segment up to the hit point ---
    // If the path is inside a medium, a scattering event may occur before the
    // surface. Homogeneous media sample the free-flight distance analytically;
    // heterogeneous media delta-track against the majorant (volume.h). On a
    // scatter the surface hit is discarded: throughput picks up the
    // sigma_s/pdf weight, NEE runs at the scatter point with the
    // phase-function closure, and the new ray is phase-sampled. On
    // pass-through the throughput picks up the transmittance weight and the
    // surface is shaded as usual.
    if (pathSegment.mediumGeom >= 0) {
        const Geom &mg = geoms[pathSegment.mediumGeom];
        const Material mm = materials[mg.material.materialId];
        Spectrum smokeA =
            upliftSigma(mm.sigmaA, pathSegment.swl) * mm.densityScale;
        Spectrum smokeS =
            upliftSigma(mm.sigmaS, pathSegment.swl) * mm.densityScale;
        Spectrum sigT = smokeA + smokeS;
        Spectrum sootA(0.0f);
        Spectrum sootS(0.0f);
        Spectrum flameA(0.0f);
        if (mg.volumeGrid.valid) {
            sootA =
                upliftSigma(mm.sootSigmaA, pathSegment.swl) * mm.densityScale;
            sootS =
                upliftSigma(mm.sootSigmaS, pathSegment.swl) * mm.densityScale;
            flameA = smokeA * mm.flameAbsorptionScale;
        }

        if (mg.volumeGrid.valid &&
            volumeSettings.debugMode >= VOLUME_DEBUG_TEMPERATURE &&
            volumeSettings.debugMode <= VOLUME_DEBUG_MAJORANT) {
            const glm::vec3 diagnostic = renderSparseVolumeDiagnostic(
                mg, mm, pathSegment.ray, intersection.t,
                volumeSettings.debugMode);
            pathSegment.radiance +=
                pathSegment.color *
                upliftIlluminant(diagnostic, pathSegment.swl, SPECTRUM_NONE,
                                 0.0f, 0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        // Estimate the emitted-radiance term independently over the complete
        // segment to the next boundary/surface:
        //   integral T(0,s) * j(s) ds.
        // This mirrors the volume rendering equation directly. Homogeneous
        // transmittance is analytic; heterogeneous transmittance is estimated
        // with ratio tracking. Keeping every carried wavelength makes the
        // estimator compatible with both packet-balance free-flight weights
        // and paths whose secondary wavelengths were terminated by dispersion.
        bool estimateSegmentEmission = true;
#if USE_MIS
        if (mg.volumeGrid.valid) {
            // Diffuse/phase vertices use the explicit emitted-volume NEE
            // hierarchy. Primary and specular paths use this ray estimator.
            estimateSegmentEmission = pathSegment.specularBounce;
            if (volumeSettings.debugMode == VOLUME_DEBUG_INDIRECT_VOLUME) {
                estimateSegmentEmission = !pathSegment.specularBounce;
            } else if (volumeSettings.debugMode ==
                           VOLUME_DEBUG_DIRECT_VOLUME ||
                       volumeSettings.debugMode ==
                           VOLUME_DEBUG_SURFACE_FIRE) {
                estimateSegmentEmission = false;
            }
        }
#endif
        if (mm.emittance > 0.0f && intersection.t > 0.0f &&
            estimateSegmentEmission) {
            if (mg.volumeGrid.valid) {
                const Spectrum emissionIntegral =
                    sparseGridEmissionIntegral(
                        mg, mm, smokeA, smokeS, sootA, sootS, flameA,
                        pathSegment.ray, intersection.t, pathSegment.swl, rng,
                        volumeStats, volumeSettings.quality);
                pathSegment.radiance +=
                    pathSegment.color * emissionIntegral;
            } else {
            const int emissionSamples =
                mm.mediumProfile == MEDIUM_PROFILE_FIRE_FRONT ? 8 : 4;
            Spectrum emissionIntegral(0.0f);
            for (int es = 0; es < emissionSamples; ++es) {
                float te =
                    ((float)es + u01(rng)) *
                    (intersection.t / emissionSamples);
                glm::vec3 emissionP =
                    pathSegment.ray.origin + te * pathSegment.ray.direction;
                float source, relativeTemperature;
                mediumEmissionProperties(mm, mg, emissionP, source,
                                         relativeTemperature);
                if (source <= 0.0f) {
                    continue;
                }
                float temperature = mm.blackbodyTemp;
                if (mm.spectrumType == SPECTRUM_BLACKBODY) {
                    temperature *= relativeTemperature;
                }
                Spectrum Le = upliftIlluminant(
                    mm.color * mm.emittance, pathSegment.swl, mm.spectrumType,
                    temperature, mm.blackbodyNorm);
                Spectrum Tr =
                    mm.heterogeneous
                        ? mediumTransmittance(mg, mm, sigT, pathSegment.ray,
                                              0.0f, te, rng)
                        : glm::exp(-sigT * te);
                emissionIntegral += Tr * Le * source;
            }
            pathSegment.radiance +=
                pathSegment.color * emissionIntegral *
                (intersection.t / (float)emissionSamples);
            }
        }

        MediumSample msamp;
        if (mg.volumeGrid.valid) {
            msamp = sampleMediumSparseGrid(
                mg, smokeA, smokeS, sootA, sootS, flameA, pathSegment.ray,
                intersection.t, rng, volumeStats);
        } else if (mm.heterogeneous) {
            msamp = sampleMediumHeterogeneous(
                mg, mm, sigT, smokeS, pathSegment.ray, intersection.t, rng);
        } else {
            msamp = sampleMediumHomogeneous(sigT, smokeS, intersection.t,
                                            u01(rng));
        }

        if (mg.volumeGrid.valid &&
            (volumeSettings.debugMode == VOLUME_DEBUG_NULL_RATE ||
             volumeSettings.debugMode == VOLUME_DEBUG_EVENT_COUNT)) {
            float value =
                volumeSettings.debugMode == VOLUME_DEBUG_NULL_RATE
                    ? (float)msamp.nullCollisions /
                          (float)fmaxf(msamp.nullCollisions +
                                         msamp.brickVisits,
                                     1)
                    : 1.0f - expf(-0.18f *
                                  (float)(msamp.nullCollisions +
                                          (msamp.scattered ? 1 : 0)));
            const glm::vec3 diagnostic =
                volumeDebugRamp(glm::clamp(value, 0.0f, 1.0f));
            pathSegment.radiance +=
                pathSegment.color *
                upliftIlluminant(diagnostic, pathSegment.swl, SPECTRUM_NONE,
                                 0.0f, 0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        pathSegment.color *= msamp.weight;
        if (msamp.scattered) {
            const int scatterLimit =
                volumeSettings.quality == VOLUME_QUALITY_DEBUG
                    ? 1
                    : volumeSettings.maxScatteringDepth;
            if (pathSegment.volumeScatteringDepth >= scatterLimit) {
                pathSegment.remainingBounces = 0;
                return;
            }
            glm::vec3 scatterP =
                pathSegment.ray.origin + msamp.t * pathSegment.ray.direction;
            glm::vec3 woW = -pathSegment.ray.direction;
            BSDF phase = makePhaseBSDF(mm.hgG);
#if USE_MIS
            sampleDirectLighting(pathSegment, phase, scatterP,
                                 /*isSurface=*/false, woW, glm::vec3(0.0f),
                                 envMap, geoms, geoms_size, materials, lights,
                                 numLights, areaLightCdf, deltaLights,
                                 numDeltaLights, rng, volumeStats,
                                 volumeSettings);
#endif
            glm::vec2 uPhase(u01(rng), u01(rng));
            BSDFSample ps;
            phase.sample(woW, uPhase, pathSegment.swl, ps);
            pathSegment.ray.origin = scatterP;
            pathSegment.ray.direction = ps.wiW;
            pathSegment.bsdfPdf = ps.pdf;
            pathSegment.specularBounce = false;
            pathSegment.lastVertexDistance = 0.0f;
            pathSegment.volumeScatteringDepth++;
            // mediumGeom unchanged: the path is still inside the medium.

#if (USE_RUSSIAN_ROULETTE)
            if (depth > 3) {
                float q = fminf(maxComponent(msamp.weight), 0.99f);
                if (q < u01(rng)) {
                    pathSegment.remainingBounces = 0;
                    return;
                }
                pathSegment.color /= q;
            }
#endif
            pathSegment.remainingBounces--;
            return;
        }
    }

    Material material = materials[intersection.materials.materialId];

    // --- Medium boundary: a null interface, not a surface ---
    // Crossing it only toggles the path's inside-a-medium state and continues
    // the ray straight through, nudged past the boundary. Not a scattering
    // event: the MIS state (bsdfPdf/specularBounce) still describes the last
    // real bounce, and no path bounce is consumed.
    if (material.type == MatType::MEDIUM) {
        bool entering = pathSegment.mediumGeom != intersection.hitGeomIndex;
        pathSegment.mediumGeom = entering ? intersection.hitGeomIndex : -1;
        const Geom &mediumGeom = geoms[intersection.hitGeomIndex];
        const glm::vec3 localDirection = glm::vec3(
            mediumGeom.transform.inverseTransform *
            glm::vec4(pathSegment.ray.direction, 0.0f));
        const float localUnitsPerWorld =
            fmaxf(glm::length(localDirection), 1e-7f);
        // Advance by about 1e-3 in the medium's local space. A fixed world
        // epsilon is too small after a large inverse scale (the wildfire
        // volume is 20 units deep) and can immediately re-hit the entrance,
        // toggling the ray back to vacuum before transport is evaluated.
        const float boundaryNudge =
            glm::clamp(1e-3f / localUnitsPerWorld, 2e-3f, 5e-2f);
        pathSegment.lastVertexDistance +=
            intersection.t + boundaryNudge;
        pathSegment.ray.origin =
            getPointOnRay(pathSegment.ray, intersection.t) +
            pathSegment.ray.direction * boundaryNudge;
        return;
    }

    glm::vec2 uv = intersection.uv;
    TextureValues texVals;

    bool hasAlbedoTexture = intersection.materials.albedoTextureID != -1;
    bool hasNormalTexture = intersection.materials.normalTextureID != -1;
    bool hasBumpTexture = intersection.materials.bumpTextureID != -1;

    texVals.albedo = glm::vec4(INFINITY);
    texVals.normal = glm::vec4(INFINITY);
    texVals.bump = glm::vec4(INFINITY);

#if USE_CHECKERBOARD_TEXTURE
    if (hasAlbedoTexture) {
        texVals.albedo = glm::vec4(checkerboard(uv.x, uv.y, 101), 1.0f);
    }
#else
    if (hasAlbedoTexture) {
        texVals.albedo = sampleTexture(
            albedoTextures[intersection.materials.albedoTextureID], uv);
    }
#endif

    // TODO: Is there a way to remove the if checks here? If not, how to make
    // this at least continuous in memory? Note that ShadeableIntersection
    // intersection is already sorted so maybe we can create a new material type
    // for each combination of textures used? But sorting has a cost itself.
    if (hasNormalTexture) {
        // TODO: You forgot to convert the normals using TBN matrix
        // https://learnopengl.com/Advanced-Lighting/Normal-Mapping
        texVals.normal = sampleTexture(
            normalTextures[intersection.materials.normalTextureID], uv);
    }

    // TODO: (same thing here)
    if (hasBumpTexture) {
        texVals.bump =
            sampleBump(bumpTextures[intersection.materials.bumpTextureID], uv);
    }

    glm::vec3 materialColor = material.color;

    // if we hit a light
    if (material.emittance > 0.0f) {
        // A BSDF-sampled ray landed on an emitter. This is one of the two MIS
        // strategies for area lights (the other is area-light NEE below), so
        // weight it against the pdf that NEE would have used to sample this
        // same direction. Full weight when NEE couldn't have taken it: a
        // specular/ primary ray, an unsupported (area == 0) emitter, or no
        // lights.
        Spectrum Le =
            upliftIlluminant(materialColor * material.emittance,
                             pathSegment.swl, material.spectrumType,
                             material.blackbodyTemp, material.blackbodyNorm);
        float weight = 1.0f;
#if USE_MIS
        if (!pathSegment.specularBounce && numLights > 0 &&
            intersection.hitGeomIndex >= 0) {
            const Geom &hitGeom = geoms[intersection.hitGeomIndex];
            const glm::vec3 hitPoint =
                getPointOnRay(pathSegment.ray, intersection.t);
            const float pdfArea =
                lightGeomPdfAreaAtPoint(hitGeom, hitPoint);
            const float selectionPmf = hitGeom.areaLightSelectionPmf;
            glm::vec3 nL = glm::normalize(intersection.surfaceGeometricNormal);
            float cosL = fabsf(glm::dot(nL, pathSegment.ray.direction));
            if (pdfArea > 0.0f && selectionPmf > 0.0f && cosL > 0.0f) {
                // Solid-angle light pdf for this exact hit, including the
                // power-weighted discrete selection PMF and (for affine
                // spheres) the local-to-world area Jacobian. This matches the
                // NEE sampler above.
                float fullDistance =
                    pathSegment.lastVertexDistance + intersection.t;
                float lightPdf = selectionPmf * pdfArea *
                                 (fullDistance * fullDistance) / cosL;
                weight = powerHeuristic(pathSegment.bsdfPdf, lightPdf);
            }
        }
#endif
        pathSegment.radiance += pathSegment.color * Le * weight;
        pathSegment.remainingBounces = 0;
        pathSegment.hasHitLight = true;
    } else {
        glm::vec3 oldIntersect = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 surfaceNormal = glm::normalize(intersection.surfaceNormal);
        glm::vec3 surfaceTangent = intersection.surfaceTangent;
        glm::vec3 woW = -pathSegment.ray.direction;

        // Geometric (face) normal, kept on the same side as the shading
        // normal (guards against inconsistent triangle winding).
        glm::vec3 Ng = glm::normalize(intersection.surfaceGeometricNormal);
        if (glm::dot(Ng, surfaceNormal) < 0.0f) {
            Ng = -Ng;
        }

        // Resolve the surface's reflectance inputs once for every BSDF call
        // below (the NEE evaluations and the scatter). An albedo texture
        // overrides the material's base color. In SPECTRAL builds the
        // resolved RGB values are uplifted to spectra here -- the single point
        // where reflectance RGB enters spectrum-land.
        glm::vec3 rgbAlbedo = materialColor;
        if (texVals.albedo != glm::vec4(INFINITY)) {
            rgbAlbedo = glm::vec3(texVals.albedo);
        }
        Spectrum albedo = upliftReflectance(rgbAlbedo, pathSegment.swl);
        Spectrum specColor =
            upliftReflectance(material.specularColor, pathSegment.swl);

        // Build the BSDF closure for this hit: face-forwarding, TBN and
        // normal/bump mapping happen once here; NEE evaluation and scattering
        // below share the resulting frame, so MIS pdfs are consistent.
        BSDF bsdf = makeBSDF(material, albedo, specColor, surfaceNormal,
                             surfaceTangent, texVals, woW);
        // Uniform, flags-derived gating (replaces the ad-hoc material-enum
        // check): delta-only closures cannot be NEE'd and take full MIS
        // weight when a BSDF ray finds a light through them.
        bool isSpecular = isDeltaOnly(bsdf.flags());

#if USE_MIS
        // Next-event estimation toward all light types (environment, area,
        // delta), shared with medium scatter events -- see
        // sampleDirectLighting above. Skipped for delta-only closures (their
        // f cannot be evaluated for a given direction; the BSDF-sampled ray
        // finds lights at full MIS weight instead).
        if (!isSpecular) {
            sampleDirectLighting(pathSegment, bsdf, oldIntersect,
                                 /*isSurface=*/true, woW, Ng, envMap, geoms,
                                 geoms_size, materials, lights, numLights,
                                 areaLightCdf, deltaLights, numDeltaLights, rng,
                                 volumeStats, volumeSettings);
        }
#endif // USE_MIS

        // Sample the BSDF for the bounce direction. Lobe selection is folded
        // into the 2D sample inside the closure, exactly as before.
        glm::vec2 uScatter(u01(rng), u01(rng));
        BSDFSample bs;
        bsdf.sample(woW, uScatter, pathSegment.swl, bs);
        glm::vec3 wiW = bs.wiW;

        // Record MIS state for the ray we're about to spawn: the env seen
        // through it (on escape) will be weighted against this pdf, unless the
        // sampled lobe was specular/delta (then it takes full weight).
        pathSegment.bsdfPdf = bs.pdf;
        pathSegment.specularBounce = hasSpecular(bs.flags);

        // Shadow-terminator fix. At grazing/silhouette angles the smooth
        // shading normal tilts away from the real facet, so a cosine sample
        // around it can point BELOW the geometric surface. Such a bounce ray
        // immediately goes into the mesh and self-occludes, leaving a dark rim
        // along silhouettes (e.g. the duck's head edge). For reflective lobes,
        // fold any below-horizon direction back above the geometric tangent
        // plane. Transmission legitimately goes below, so any closure with a
        // transmissive lobe is left alone (flags-based, was a dielectric enum
        // check).
        if (!hasTransmission(bsdf.flags()) && glm::dot(wiW, Ng) < 0.0f) {
            wiW = glm::normalize(wiW - 2.0f * glm::dot(wiW, Ng) * Ng);
        }

        pathSegment.ray.direction = wiW; // wiW should already be normalized
        // Offset the new origin along the GEOMETRIC normal (it follows the real
        // facet) on whichever side wiW leaves, so it reliably clears the
        // surface even at grazing angles -- offsetting along the shading normal
        // or along wiW does not, which is what produced the dark edge.
        glm::vec3 offsetNormal = glm::dot(wiW, Ng) < 0.0f ? -Ng : Ng;
        pathSegment.ray.origin = oldIntersect + offsetNormal * 1e-3f;
        pathSegment.lastVertexDistance = 0.0f;
        pathSegment.color *= bs.weight;

// TODO: is it worth it?
#if (USE_RUSSIAN_ROULETTE) // Possibly terminate the path with Russian roulette
        if (depth > 3) {
            // So that the ray can bounce for a bit before we start terminating
            // it. In SPECTRAL builds this spans all carried wavelengths
            // (terminated secondaries are zero, so the hero drives survival).
            float maxThroughput = maxComponent(bs.weight);
            float survivalProbability = u01(rng);
            float eta_sq = bs.eta * bs.eta;
            float q = fminf(maxThroughput * eta_sq, 0.99f);

            if (q < survivalProbability) {
                pathSegment.remainingBounces = 0;
                return;
            } else {
                pathSegment.color /= q;
            }
        }
#endif

        pathSegment.remainingBounces--;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image,
                            PathSegment *iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths) {
        PathSegment iterationPath = iterationPaths[index];
        // `radiance` is the path's accumulated light (emitter/env hits + NEE).
        // Add it unconditionally: with next-event estimation a path can carry
        // radiance even if it never terminated on a light itself.
        // SPECTRAL builds convert the spectral estimate to RGB here (the only
        // exit from spectrum-land); dev.image stays glm::vec3 either way.
        glm::vec3 c = spectrumToRGB(iterationPath.radiance, iterationPath.swl);
        // Drop non-finite samples. A NaN/Inf from a degenerate BSDF or
        // frame (e.g. a divide-by-zero in a near-mirror microfacet lobe, or
        // normalize() of a zero half-vector) would otherwise be added once
        // and poison the pixel for the whole render -> a permanent speckle.
        if (isfinite(c.x) && isfinite(c.y) && isfinite(c.z)) {
#if USE_FIREFLY_CLAMP
            // Scale down (preserving hue) any sample brighter than the cap.
            float m = fmaxf(c.x, fmaxf(c.y, c.z));
            if (m > FIREFLY_CLAMP_MAX) {
                c *= FIREFLY_CLAMP_MAX / m;
            }
#endif
            image[iterationPath.pixelIndex] += c;
        }
    }
}

__global__ void
computeIsIntersected(int num_paths, int *isIntersected,
                     const ShadeableIntersection *intersections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    isIntersected[idx] = intersections[idx].t != -1.0f;
}

void partitionRays(int &num_paths, PathSegment *dev_paths,
                   const ShadeableIntersection *dev_intersections) {
    thrust::device_ptr<PathSegment> dev_ptr(dev_paths);
    // stable_partition differs from partition in that stable_partition is
    // guaranteed to preserve relative order.
    thrust::device_ptr<PathSegment> dev_ptr_end = thrust::stable_partition(
        thrust::device, dev_ptr, dev_ptr + num_paths, has_remaining_bounces());
    cudaDeviceSynchronize();
    num_paths = dev_ptr_end - dev_ptr;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    // Do not let a stale allocation accidentally re-enable instrumentation if
    // an interactive caller changes integrator settings between frames.
    VolumeTrackingStats *const volumeTrackingStats =
        hst_scene->volumeIntegrator.reportTrackingStats != 0
            ? dev.volumeTrackingStats
            : nullptr;

    // 2D block for generating ray from camera
    // This is a common choice for image workloads: a small 2D tile of pixels
    // handled by a block
    const dim3 blockSize2d(8, 8);
    // To cover N items with blocks of size B, you do (N + B - 1) / B, i.e.
    // ceil(N / B)
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented.
    //     Currently, intersection distance is recorded as a parametric
    //     distance, t, or a "distance along the ray." t = -1.0 indicates no
    //     intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, dev.paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    // pointer arithmetic is in units of elements, not bytes
    PathSegment *dev_path_end = dev.paths + pixelcount;
    int total_num_paths = dev_path_end - dev.paths;
    int num_paths = total_num_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    bool iterationComplete = false;
    // remainingBounces, decremented only at real surface/medium vertices, is
    // the actual path-depth budget. This separate generous cap only guards
    // against a numerical loop repeatedly hitting the same null boundary.
    const int maxTraversalSteps =
        traceDepth * (2 * (int)hst_scene->geoms.size() + 2) + 8;
    while (!iterationComplete) {
        // clean shading chunks
        cudaMemset(dev.intersections, 0,
                   pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing =
            (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
            depth, num_paths, dev.paths, dev.geoms, hst_scene->geoms.size(),
            dev.intersections);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

// Sort materials by type
#if USE_MATERIAL_SORT
        thrust::sort_by_key(thrust::device, dev.intersections,
                            dev.intersections + num_paths, dev.paths,
                            sortMaterialCondition());
        cudaDeviceSynchronize();
#endif

        shade<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter, depth, num_paths, dev.intersections, dev.paths, dev.materials,
            dev.albedoTextures, dev.normalTextures, dev.bumpTextures,
            dev.envMap, dev.geoms, hst_scene->geoms.size(), dev.lights,
            (int)hst_scene->lights.size(), dev.areaLightCdf, dev.deltaLights,
            (int)hst_scene->deltaLights.size(), volumeTrackingStats,
            hst_scene->volumeIntegrator);
        cudaDeviceSynchronize();

#if USE_STREAM_COMPACTION
        // compact paths
        partitionRays(num_paths, dev.paths, dev.intersections);
#endif

        iterationComplete = (num_paths == 0) || (depth >= maxTraversalSteps);

        if (guiData != NULL) {
            guiData->TracedDepth = depth;
        }
    }
    if (num_paths > 0 && depth >= maxTraversalSteps) {
        fprintf(stderr,
                "WARNING: null-interface safety cap reached with %d live "
                "paths; check medium boundaries for numerical loops.\n",
                num_paths);
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(total_num_paths, dev.image,
                                                  dev.paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    // Note this is not ping pong buffers! It's doing classic path tracer where
    // the results get average after each loop.
    if (pbo != nullptr) {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
            pbo, cam.resolution, iter, dev.image, cam.exposure, cam.toneMap);
    }

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev.image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
