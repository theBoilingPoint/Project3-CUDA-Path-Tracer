#include "pathtrace.h"

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

#include "cudaUtil.h"
#include "deviceScene.h"
#include "glm/glm.hpp"
#include "interactions.h"
#include "intersections.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

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
makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter,
                               glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

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
        makeSeededRandomEngine(iter, index, traceDepth);
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
    segment.color = glm::vec3(1.0f);    // throughput
    segment.radiance = glm::vec3(0.0f); // accumulated light
    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    segment.hasHitLight = false;
    // The camera ray has no preceding bounce; flag it specular so directly
    // viewed env/emitters are added at full weight (no MIS discount).
    segment.bsdfPdf = 0.0f;
    segment.specularBounce = true;
    segment.eta = 1.0f;
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

    // Stored as (du, dv) for the normal perturbation in scatterRay.
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
// is effectively unbounded, so any hit blocks it.
__device__ bool anyHit(const Ray &ray, Geom *geoms, int geoms_size,
                       float tMax) {
    glm::vec3 tmpP, tmpN, tmpGN, tmpT;
    glm::vec2 tmpUV;
    bool outside;
    for (int i = 0; i < geoms_size; ++i) {
        Geom &geom = geoms[i];
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

// World-space surface area of an emitter geom, from the linear part of its
// transform. Supports analytic cube and sphere emitters, and triangle meshes
// (summed per-triangle world area). Returns 0 for anything else.
//
// NOTE: the mesh case is O(numTriangles) and is evaluated per light sample /
// per emitter hit. That is fine for the low-poly emitters these scenes use; a
// large mesh light would want a precomputed area (and triangle-area CDF)
// uploaded once, like the environment distribution.
__device__ float lightGeomArea(const Geom &g) {
    glm::vec3 ex =
        glm::vec3(g.transform.transform[0]); // local +X edge in world
    glm::vec3 ey = glm::vec3(g.transform.transform[1]);
    glm::vec3 ez = glm::vec3(g.transform.transform[2]);
    if (g.type == CUBE) {
        // Unit cube [-0.5,0.5]^3: full edge vectors are ex/ey/ez, so opposite
        // face pairs give 2*(|ey x ez| + |ex x ez| + |ex x ey|).
        return 2.0f * (glm::length(glm::cross(ey, ez)) +
                       glm::length(glm::cross(ex, ez)) +
                       glm::length(glm::cross(ex, ey)));
    } else if (g.type == SPHERE) {
        // Local radius 0.5; assumes ~uniform scale (world radius 0.5*|ex|).
        float r = 0.5f * glm::length(ex);
        return 4.0f * M_PIf * r * r;
    } else if (g.type == MESH) {
        const Triangle *tris = g.geometry.devTriangles;
        int n = g.geometry.numTriangles;
        float area = 0.0f;
        for (int i = 0; i < n; ++i) {
            glm::vec3 p0 = multiplyMV(g.transform.transform,
                                      glm::vec4(tris[i].points[0], 1.0f));
            glm::vec3 p1 = multiplyMV(g.transform.transform,
                                      glm::vec4(tris[i].points[1], 1.0f));
            glm::vec3 p2 = multiplyMV(g.transform.transform,
                                      glm::vec4(tris[i].points[2], 1.0f));
            area += 0.5f * glm::length(glm::cross(p1 - p0, p2 - p0));
        }
        return area;
    }
    return 0.0f;
}

// Sample a world-space point + outward world normal uniformly on an emitter's
// surface. Outputs the area-measure pdf (1/area); 0 for unsupported geoms.
__device__ void sampleLightGeom(const Geom &g, float u1, float u2, float u3,
                                glm::vec3 &pWorld, glm::vec3 &nWorld,
                                float &pdfArea) {
    float area = lightGeomArea(g);
    pdfArea = area > 0.0f ? 1.0f / area : 0.0f;
    if (pdfArea == 0.0f) {
        return;
    }

    // Mesh emitter: pick a triangle proportional to world area, then sample a
    // uniform barycentric point on it. Area-weighted selection makes the pdf
    // uniform over the surface (1/totalArea), matching lightGeomArea used by
    // the reverse MIS weight. (Two O(numTriangles) passes -- see the note
    // above.)
    if (g.type == MESH) {
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
        // Uniform point on the unit sphere (local radius 0.5).
        float z = 1.0f - 2.0f * u1;
        float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        float phi = 2.0f * M_PIf * u2;
        nLocal = glm::vec3(r * cosf(phi), r * sinf(phi), z);
        pLocal = 0.5f * nLocal;
    } else { // CUBE
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

__device__ glm::vec3 checkerboard(float u, float v, int checkerSize) {
    int u_check = static_cast<int>(floor(u * checkerSize)) % 2;
    int v_check = static_cast<int>(floor(v * checkerSize)) % 2;

    if (u_check == v_check) {
        return glm::vec3(1.0f, 1.0f, 1.0f); // white square
    } else {
        return glm::vec3(0.0f, 0.0f, 0.0f); // black square
    }
}

__global__ void shade(int iter, int depth, int num_paths,
                      ShadeableIntersection *shadeableIntersections,
                      PathSegment *pathSegments, Material *materials,
                      Texture *albedoTextures, Texture *normalTextures,
                      Texture *bumpTextures, EnvironmentMap envMap, Geom *geoms,
                      int geoms_size, Geom *lights, int numLights,
                      DeltaLight *deltaLights, int numDeltaLights) {
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
            glm::vec3 Le = sampleEnvironment(envMap, dir);
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

    Material material = materials[intersection.materials.materialId];
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
        glm::vec3 Le = materialColor * material.emittance;
        float weight = 1.0f;
#if USE_MIS
        if (!pathSegment.specularBounce && numLights > 0 &&
            intersection.hitGeomIndex >= 0) {
            float area = lightGeomArea(geoms[intersection.hitGeomIndex]);
            glm::vec3 nL = glm::normalize(intersection.surfaceGeometricNormal);
            float cosL = fabsf(glm::dot(nL, pathSegment.ray.direction));
            if (area > 0.0f && cosL > 0.0f) {
                // Solid-angle light pdf for this hit, incl. 1/numLights uniform
                // light selection -- matches the NEE sampler below.
                float lightPdf = (intersection.t * intersection.t) /
                                 (area * cosL * (float)numLights);
                weight = powerHeuristic(pathSegment.bsdfPdf, lightPdf);
            }
        }
#endif
        pathSegment.radiance += pathSegment.color * Le * weight;
        pathSegment.remainingBounces = 0;
        pathSegment.hasHitLight = true;
    } else {
        thrust::default_random_engine rng =
            makeSeededRandomEngine(iter, idx, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        glm::vec3 oldIntersect = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 surfaceNormal = glm::normalize(intersection.surfaceNormal);
        glm::vec3 surfaceTangent = intersection.surfaceTangent;
        glm::vec3 woW = -pathSegment.ray.direction;
        glm::vec3 wiW;
        glm::vec3 c;
        float pdf;
        float eta;

        // Geometric (face) normal, kept on the same side as the shading
        // normal (guards against inconsistent triangle winding).
        glm::vec3 Ng = glm::normalize(intersection.surfaceGeometricNormal);
        if (glm::dot(Ng, surfaceNormal) < 0.0f) {
            Ng = -Ng;
        }

        bool isSpecular = (material.type == MatType::MIRROR ||
                           material.type == MatType::DIELECTRIC);

#if USE_MIS
        // --- Next-event estimation toward the environment light (MIS) ---
        // Sample a direction from the env's luminance distribution, evaluate
        // the BSDF for it, and add its (visibility-tested) contribution
        // weighted against BSDF sampling. Skipped for specular lobes (delta
        // BSDF cannot be evaluated for an arbitrary direction; the BSDF-sampled
        // escape already captures it at full weight).
        if (envMap.valid && envMap.distValid && !isSpecular) {
            glm::vec3 lightDir;
            float lightPdf;
            glm::vec3 Le = sampleEnvDirection(envMap, u01(rng), u01(rng),
                                              lightDir, lightPdf);
            if (lightPdf > 0.0f && glm::dot(lightDir, Ng) > 0.0f) {
                glm::vec3 f;
                float bsdfPdfL;
                evalBSDF(woW, surfaceNormal, surfaceTangent, lightDir, material,
                         texVals, f, bsdfPdfL);
                float cosAtSurface = glm::dot(lightDir, surfaceNormal);
                if (bsdfPdfL > 0.0f && cosAtSurface > 0.0f &&
                    (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)) {
                    Ray shadowRay;
                    shadowRay.origin = oldIntersect + Ng * 1e-3f;
                    shadowRay.direction = lightDir;
                    if (!anyHit(shadowRay, geoms, geoms_size, FLT_MAX)) {
                        float weight = powerHeuristic(lightPdf, bsdfPdfL);
                        pathSegment.radiance += pathSegment.color * f *
                                                cosAtSurface * Le * weight /
                                                lightPdf;
                    }
                }
            }
        }

        // --- Next-event estimation toward an area light (MIS) ---
        // Pick one emitter uniformly, sample a point on it, and add its
        // shadow-tested contribution weighted against BSDF sampling. Dividing
        // by the 1/numLights selection probability makes this an unbiased
        // estimate of all the area lights' direct contribution.
        if (numLights > 0 && !isSpecular) {
            int li = min((int)(u01(rng) * numLights), numLights - 1);
            Geom L = lights[li];
            glm::vec3 pL, nL;
            float pdfArea;
            sampleLightGeom(L, u01(rng), u01(rng), u01(rng), pL, nL, pdfArea);
            if (pdfArea > 0.0f) {
                glm::vec3 d = pL - oldIntersect;
                float dist = sqrtf(glm::dot(d, d));
                glm::vec3 lightDir = d / dist;
                float pdfSA =
                    lightPdfSolidAngle(pdfArea, oldIntersect, pL, nL) /
                    (float)numLights;
                float cosAtSurface = glm::dot(lightDir, surfaceNormal);
                if (pdfSA > 0.0f && cosAtSurface > 0.0f &&
                    glm::dot(lightDir, Ng) > 0.0f) {
                    glm::vec3 f;
                    float bsdfPdfL;
                    evalBSDF(woW, surfaceNormal, surfaceTangent, lightDir,
                             material, texVals, f, bsdfPdfL);
                    if (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f) {
                        Ray shadowRay;
                        shadowRay.origin = oldIntersect + Ng * 1e-3f;
                        shadowRay.direction = lightDir;
                        // Stop just short of the light so its own surface does
                        // not count as an occluder.
                        if (!anyHit(shadowRay, geoms, geoms_size,
                                    dist - 1e-3f)) {
                            Material lMat = materials[L.material.materialId];
                            glm::vec3 Le = lMat.color * lMat.emittance;
                            float weight = powerHeuristic(pdfSA, bsdfPdfL);
                            pathSegment.radiance += pathSegment.color * f *
                                                    cosAtSurface * Le * weight /
                                                    pdfSA;
                        }
                    }
                }
            }
        }

        // --- Next-event estimation toward delta (point/directional) lights ---
        // Delta lights can't be hit by BSDF sampling, so each is a single
        // shadow-ray sample with no MIS weight (weight = 1).
        for (int li = 0; li < numDeltaLights && !isSpecular; ++li) {
            DeltaLight dl = deltaLights[li];
            glm::vec3 lightDir;
            float dist;
            glm::vec3 Li;
            if (dl.type == POINT_LIGHT) {
                glm::vec3 d = dl.position - oldIntersect;
                float dist2 = glm::dot(d, d);
                dist = sqrtf(dist2);
                lightDir = d / dist;
                Li = dl.radiance / dist2; // inverse-square falloff
            } else {                      // DIRECTIONAL_LIGHT
                lightDir = -glm::normalize(dl.direction);
                dist = FLT_MAX;
                Li = dl.radiance;
            }
            float cosAtSurface = glm::dot(lightDir, surfaceNormal);
            if (cosAtSurface <= 0.0f || glm::dot(lightDir, Ng) <= 0.0f) {
                continue;
            }
            glm::vec3 f;
            float bsdfPdfL;
            evalBSDF(woW, surfaceNormal, surfaceTangent, lightDir, material,
                     texVals, f, bsdfPdfL);
            if (f.x <= 0.0f && f.y <= 0.0f && f.z <= 0.0f) {
                continue;
            }
            Ray shadowRay;
            shadowRay.origin = oldIntersect + Ng * 1e-3f;
            shadowRay.direction = lightDir;
            float tMax = (dl.type == POINT_LIGHT) ? dist - 1e-3f : FLT_MAX;
            if (!anyHit(shadowRay, geoms, geoms_size, tMax)) {
                pathSegment.radiance +=
                    pathSegment.color * f * cosAtSurface * Li;
            }
        }
#endif // USE_MIS

        scatterRay(pathSegment, woW, surfaceNormal, surfaceTangent, wiW, pdf, c,
                   eta, material, texVals, rng);

        // Record MIS state for the ray we're about to spawn: the env seen
        // through it (on escape) will be weighted against this pdf, unless the
        // bounce was specular (then it takes full weight).
        pathSegment.bsdfPdf = pdf;
        pathSegment.specularBounce = isSpecular;

        // Shadow-terminator fix. At grazing/silhouette angles the smooth
        // shading normal tilts away from the real facet, so a cosine sample
        // around it can point BELOW the geometric surface. Such a bounce ray
        // immediately goes into the mesh and self-occludes, leaving a dark rim
        // along silhouettes (e.g. the duck's head edge). For reflective lobes,
        // fold any below-horizon direction back above the geometric tangent
        // plane. Transmission legitimately goes below, so leave dielectric be.
        if (material.type != MatType::DIELECTRIC && glm::dot(wiW, Ng) < 0.0f) {
            wiW = glm::normalize(wiW - 2.0f * glm::dot(wiW, Ng) * Ng);
        }

        pathSegment.ray.direction = wiW; // wiW should already be normalized
        // Offset the new origin along the GEOMETRIC normal (it follows the real
        // facet) on whichever side wiW leaves, so it reliably clears the
        // surface even at grazing angles -- offsetting along the shading normal
        // or along wiW does not, which is what produced the dark edge.
        glm::vec3 offsetNormal = glm::dot(wiW, Ng) < 0.0f ? -Ng : Ng;
        pathSegment.ray.origin = oldIntersect + offsetNormal * 1e-3f;
        pathSegment.color *= c;

// TODO: is it worth it?
#if (USE_RUSSIAN_ROULETTE) // Possibly terminate the path with Russian roulette
        if (depth > 3) {
            // So that the ray can bounce for a bit before we start terminating
            // it
            float maxComponent = fmaxf(c.x, fmaxf(c.y, c.z));
            float survivalProbability = u01(rng);
            float eta_sq = eta * eta;
            float q = fminf(maxComponent * eta_sq, 0.99f);

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
        glm::vec3 c = iterationPath.radiance;
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
            (int)hst_scene->lights.size(), dev.deltaLights,
            (int)hst_scene->deltaLights.size());
        cudaDeviceSynchronize();

#if USE_STREAM_COMPACTION
        // compact paths
        partitionRays(num_paths, dev.paths, dev.intersections);
#endif

        iterationComplete = (depth >= traceDepth) || (num_paths == 0);

        if (guiData != NULL) {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(total_num_paths, dev.image,
                                                  dev.paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    // Note this is not ping pong buffers! It's doing classic path tracer where
    // the results get average after each loop.
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
                                                     dev.image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev.image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}