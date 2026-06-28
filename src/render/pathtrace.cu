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
    segment.color = glm::vec3(1.0f);
    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    segment.hasHitLight = false;
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
        glm::vec3 tangent;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
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
            } else if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                           tmp_normal, outside);
            } else if (geom.type == MESH) {
#if USE_BVH
                t = meshIntersectionTestBVH(geom, pathSegment.ray,
                                            tmp_intersect, tmp_normal,
                                            tmp_tangent, tmp_uv, outside);
#else
                t = meshIntersectionTestNaive(geom, pathSegment.ray,
                                              tmp_intersect, tmp_normal,
                                              tmp_tangent, tmp_uv, outside);
#endif
            }

            // Compute the minimum t from the intersection tests to determine
            // what scene geometry object was hit first.
            if (t > 0.0f && t_min > t) {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                tangent = tmp_tangent;
                uv = tmp_uv;
            }
        }

        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
        } else {
            Geom hitGeom = geoms[hit_geom_index];
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materials = hitGeom.material;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].surfaceTangent = tangent;
            intersections[path_index].uv = uv;
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
        dir = glm::vec3(c * dir.x + s * dir.z, dir.y,
                        -s * dir.x + c * dir.z);
    }

    float u = 0.5f + atan2f(dir.z, dir.x) * (0.5f * M_1_PIf);
    float v = 0.5f - asinf(glm::clamp(dir.y, -1.0f, 1.0f)) * M_1_PIf;
    float4 t = tex2D<float4>(env.texObj, u, v);
    return glm::vec3(t.x, t.y, t.z) * env.intensity;
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
                      Texture *bumpTextures, EnvironmentMap envMap) {
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
        // Ray escaped the scene. Sample the environment map (if any) in the
        // ray's direction and treat it as incoming radiance: pathSegment.color
        // is the accumulated throughput, so this is the background for primary
        // rays and image-based lighting for bounced rays.
        if (envMap.valid) {
            pathSegment.color *=
                sampleEnvironment(envMap, pathSegment.ray.direction);
            pathSegment.hasHitLight = true;
        } else {
            pathSegment.color = glm::vec3(0.0f);
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
        pathSegment.color *= materialColor * material.emittance;
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

        scatterRay(pathSegment, woW, surfaceNormal, surfaceTangent, wiW, pdf, c,
                   eta, material, texVals, rng);

        pathSegment.ray.direction = wiW; // wiW should already be normalized
        // Without the offset, when the ray immediately intersects the surface
        // it originated from, the refraction calculations may fail or yield
        // invalid results, such as: Total Internal Reflection: The refracted
        // ray might get treated as a reflective ray due to intersection
        // problems, resulting in no transmitted light. Black Pixels: The lack
        // of refraction or valid light contribution can result in areas
        // appearing black.
        pathSegment.ray.origin =
            oldIntersect + pathSegment.ray.direction * 0.01f;
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
        if (iterationPath.hasHitLight) {
            image[iterationPath.pixelIndex] += iterationPath.color;
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
            dev.envMap);
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