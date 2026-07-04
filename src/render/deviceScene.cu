#include "deviceScene.h"

#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "cudaUtil.h"

// Uploads each mesh geom's triangles and flattened BVH nodes to the device,
// concatenated into one big array each, and points the per-geom device pointers
// into them. The device pointers are taken by reference so the caller's members
// are set (and thus freeable).
static void initialiseTriangles(Triangle *&dev_triangles,
                                LinearBVHNode *&dev_nodes,
                                std::vector<Geom> &geometries,
                                std::vector<MeshData> &meshData,
                                const int totalNumberOfGeom) {
    if (totalNumberOfGeom == 0) {
        return;
    }

    int totalNumberOfTriangles = 0;
    int totalNumberOfNodes = 0;
    for (int i = 0; i < totalNumberOfGeom; i++) {
        if (geometries[i].type == MESH) {
            totalNumberOfTriangles += meshData[i].numTriangles;
            totalNumberOfNodes += meshData[i].numNodes;
        }
    }

    if (totalNumberOfTriangles == 0) {
        return;
    }

    cudaMalloc(&dev_triangles, totalNumberOfTriangles * sizeof(Triangle));
    cudaMalloc(&dev_nodes, totalNumberOfNodes * sizeof(LinearBVHNode));
    int triOffset = 0;
    int nodeOffset = 0;
    for (int i = 0; i < totalNumberOfGeom; i++) {
        if (geometries[i].type == MESH) {
            // Copy each geometry's triangles to the device memory
            cudaMemcpy(dev_triangles + triOffset, meshData[i].triangles,
                       meshData[i].numTriangles * sizeof(Triangle),
                       cudaMemcpyHostToDevice);
            geometries[i].geometry.devTriangles = dev_triangles + triOffset;
            triOffset += meshData[i].numTriangles;

            // Copy each geometry's flattened BVH nodes to the device memory.
            // Leaf primitivesOffset / interior secondChildOffset are relative
            // to this mesh's own arrays, so the per-mesh base pointer is
            // correct.
            cudaMemcpy(dev_nodes + nodeOffset, meshData[i].nodes,
                       meshData[i].numNodes * sizeof(LinearBVHNode),
                       cudaMemcpyHostToDevice);
            geometries[i].geometry.devNodes = dev_nodes + nodeOffset;
            nodeOffset += meshData[i].numNodes;
        }
    }
}

// Upload one texture into a cudaArray and wrap it in a hardware texture object.
// glm::vec4 is bit-compatible with float4, so the host data copies straight in.
// Records the texture object + backing array in ds so teardown can release
// them; the device only ever needs the handle.
static Texture createTextureObject(DeviceScene &ds, const glm::vec4 *hostData,
                                   glm::ivec2 size,
                                   cudaTextureFilterMode filterMode,
                                   cudaTextureAddressMode addressMode,
                                   unsigned int arrayFlags = 0) {
    // Allocate an array-backed (not linear) allocation; hardware filtering and
    // the texture cache require array storage. arrayFlags lets callers request
    // cudaArrayTextureGather, which tex2Dgather (used for bump maps) requires.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, size.x, size.y, arrayFlags);

    const size_t rowBytes = size.x * sizeof(float4);
    cudaMemcpy2DToArray(cuArray, 0, 0, hostData, rowBytes, rowBytes, size.y,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.readMode = cudaReadModeElementType; // keep float values as-is
    texDesc.normalizedCoords = 1;               // sample with UVs in [0, 1]

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    ds.textureResources.push_back({texObj, cuArray});

    Texture tex;
    tex.texObj = texObj;
    return tex;
}

static void copyTexturesFromHostToDevice(
    DeviceScene &ds, const int numTextures,
    const std::vector<std::tuple<glm::vec4 *, glm::ivec2>> &textures,
    Texture *&dev_textures, cudaTextureFilterMode filterMode,
    cudaTextureAddressMode addressMode, unsigned int arrayFlags = 0) {
    // Allocate the device-side Texture array (each entry just holds a texture
    // object handle).
    cudaMalloc(&dev_textures, numTextures * sizeof(Texture));

    std::vector<Texture> h_textures(numTextures);
    for (int i = 0; i < numTextures; i++) {
        glm::vec4 *hostTextureData = std::get<0>(textures[i]);
        glm::ivec2 textureSize = std::get<1>(textures[i]);
        h_textures[i] =
            createTextureObject(ds, hostTextureData, textureSize, filterMode,
                                addressMode, arrayFlags);
    }

    cudaMemcpy(dev_textures, h_textures.data(), numTextures * sizeof(Texture),
               cudaMemcpyHostToDevice);

    checkCUDAError("Texture Copying");
}

static void initialiseTextures(DeviceScene &ds, Scene *scene) {
    std::vector<std::tuple<glm::vec4 *, glm::ivec2>> &albedoTextures =
        scene->albedoTextures;
    std::vector<std::tuple<glm::vec4 *, glm::ivec2>> &normalTextures =
        scene->normalTextures;
    std::vector<std::tuple<glm::vec4 *, glm::ivec2>> &bumpTextures =
        scene->bumpTextures;

    if (albedoTextures.size() > 0) {
        // Albedo: bilinear filtering + wrap addressing (standard for tiled
        // color maps).
        copyTexturesFromHostToDevice(ds, albedoTextures.size(), albedoTextures,
                                     ds.albedoTextures, cudaFilterModeLinear,
                                     cudaAddressModeWrap);
        checkCUDAError("Alebdo Textures Initialisation");
    }

    if (normalTextures.size() > 0) {
        // Normal maps: bilinear filtering of tangent-space normals is the
        // standard approach.
        copyTexturesFromHostToDevice(ds, normalTextures.size(), normalTextures,
                                     ds.normalTextures, cudaFilterModeLinear,
                                     cudaAddressModeWrap);
        checkCUDAError("Normal Textures Initialisation");
    }

    if (bumpTextures.size() > 0) {
        // Bump/height maps: clamp at edges. cudaArrayTextureGather lets us read
        // the whole 2x2 height neighborhood in one tex2Dgather op for the
        // finite difference. (Filter mode is irrelevant for gather.)
        copyTexturesFromHostToDevice(
            ds, bumpTextures.size(), bumpTextures, ds.bumpTextures,
            cudaFilterModePoint, cudaAddressModeClamp, cudaArrayTextureGather);
        checkCUDAError("Bump Textures Initialisation");
    }

    checkCUDAError("Texture Initialisation");
}

// Upload the equirectangular HDR environment map (if any) to a CUDA texture
// object. Bilinear filtering smooths the lat-long lookup; wrap addressing is
// correct for the azimuth (U) seam and harmless at the poles (V stays in
// [0, 1]). The backing array/handle are tracked in ds.textureResources, so
// teardown is handled by freeTextureResources.
static void initialiseEnvironmentMap(DeviceScene &ds, Scene *scene) {
    if (!scene->hasEnvMap || scene->envMap == nullptr) {
        ds.envMap.valid = 0;
        return;
    }

    Texture tex =
        createTextureObject(ds, scene->envMap, scene->envMapSize,
                            cudaFilterModeLinear, cudaAddressModeWrap);
    ds.envMap.texObj = tex.texObj;
    ds.envMap.intensity = scene->envMapIntensity;
    ds.envMap.rotation = scene->envMapRotation;
    ds.envMap.valid = 1;
    ds.envMap.width = scene->envMapSize.x;
    ds.envMap.height = scene->envMapSize.y;
    ds.envMap.conditionalCdf = nullptr;
    ds.envMap.marginalCdf = nullptr;
    ds.envMap.distValid = 0;

    // Upload the importance-sampling CDFs (for NEE + MIS), if they were built.
    if (!scene->envConditionalCdf.empty() && !scene->envMarginalCdf.empty()) {
        float *devConditional = nullptr;
        float *devMarginal = nullptr;
        const size_t condBytes =
            scene->envConditionalCdf.size() * sizeof(float);
        const size_t margBytes = scene->envMarginalCdf.size() * sizeof(float);
        cudaMalloc(&devConditional, condBytes);
        cudaMalloc(&devMarginal, margBytes);
        cudaMemcpy(devConditional, scene->envConditionalCdf.data(), condBytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(devMarginal, scene->envMarginalCdf.data(), margBytes,
                   cudaMemcpyHostToDevice);
        ds.envMap.conditionalCdf = devConditional;
        ds.envMap.marginalCdf = devMarginal;
        ds.envMap.distValid = 1;
    }

    checkCUDAError("Environment Map Initialisation");
}

void deviceSceneInit(DeviceScene &ds, Scene *scene) {
    const Camera &cam = scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&ds.image, pixelcount * sizeof(glm::vec3));
    cudaMemset(ds.image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&ds.paths, pixelcount * sizeof(PathSegment));

    int totalNumberOfGeom = scene->geoms.size();
    initialiseTriangles(
        ds.geomTriangles, ds.geomBVHNodes, scene->geoms, scene->geomMeshData,
        totalNumberOfGeom); // Must appear before initializing ds.geoms
    cudaMalloc(&ds.geoms, totalNumberOfGeom * sizeof(Geom));
    cudaMemcpy(ds.geoms, scene->geoms.data(),
               scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    int totalNumberOfLights = scene->lights.size();
    initialiseTriangles(
        ds.lightTriangles, ds.lightBVHNodes, scene->lights,
        scene->lightMeshData,
        totalNumberOfLights); // Must appear before initializing ds.lights
    cudaMalloc(&ds.lights, totalNumberOfLights * sizeof(Geom));
    cudaMemcpy(ds.lights, scene->lights.data(),
               scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    cudaMalloc(&ds.totalNumberOfLights, sizeof(int));
    cudaMemcpy(ds.totalNumberOfLights, &totalNumberOfLights, sizeof(int),
               cudaMemcpyHostToDevice);

    // Delta (point/directional) lights.
    ds.deltaLights = nullptr;
    if (!scene->deltaLights.empty()) {
        cudaMalloc(&ds.deltaLights,
                   scene->deltaLights.size() * sizeof(DeltaLight));
        cudaMemcpy(ds.deltaLights, scene->deltaLights.data(),
                   scene->deltaLights.size() * sizeof(DeltaLight),
                   cudaMemcpyHostToDevice);
    }

    // NOTE: the host-side triangle/BVH arrays in scene->geomMeshData are
    // intentionally NOT freed here. deviceSceneInit re-runs on every camera
    // change (runCuda -> pathtraceFree + pathtraceInit when iteration resets to
    // 0), and that re-upload reads these same host arrays. Freeing them here
    // left dangling/null pointers, so the next re-init issued a HostToDevice
    // cudaMemcpy from null and failed with cudaErrorInvalidValue (surfaced
    // stickily at a later checkCUDAError). Scene owns this memory and frees it
    // once in ~Scene().

    initialiseTextures(ds, scene);
    initialiseEnvironmentMap(ds, scene);

    cudaMalloc(&ds.materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(ds.materials, scene->materials.data(),
               scene->materials.size() * sizeof(Material),
               cudaMemcpyHostToDevice);

    cudaMalloc(&ds.intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(ds.intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("deviceSceneInit");
}

// Destroy every texture object and free its backing cudaArray. Resources are
// tracked host-side (see TextureResource), so no device->host copy is needed.
static void freeTextureResources(DeviceScene &ds) {
    for (const TextureResource &r : ds.textureResources) {
        if (r.texObj != 0) {
            cudaDestroyTextureObject(r.texObj);
        }
        if (r.array != nullptr) {
            cudaFreeArray(r.array);
        }
    }
    ds.textureResources.clear();
}

void deviceSceneFree(DeviceScene &ds) {
    cudaFree(ds.image); // no-op if null
    cudaFree(ds.paths);
    cudaFree(ds.geoms);
    cudaFree(ds.lights);
    cudaFree(ds.geomTriangles);
    cudaFree(ds.lightTriangles);
    cudaFree(ds.geomBVHNodes);
    cudaFree(ds.lightBVHNodes);
    cudaFree(ds.totalNumberOfLights);
    cudaFree(ds.deltaLights); // no-op if null
    cudaFree(ds.materials);
    cudaFree(ds.intersections);

    // Destroy the texture objects + backing arrays (tracked host-side), then
    // free the device-side handle arrays themselves.
    freeTextureResources(ds);
    cudaFree(ds.albedoTextures); // no-op if null
    cudaFree(ds.normalTextures);
    cudaFree(ds.bumpTextures);

    // Env importance-sampling CDFs (const pointers; cast away for cudaFree).
    cudaFree((void *)ds.envMap.conditionalCdf);
    cudaFree((void *)ds.envMap.marginalCdf);
    ds.envMap.conditionalCdf = nullptr;
    ds.envMap.marginalCdf = nullptr;

    checkCUDAError("deviceSceneFree");
}
