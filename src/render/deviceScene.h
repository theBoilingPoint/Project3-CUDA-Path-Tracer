#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "scene.h"        // Scene
#include "sceneStructs.h" // Geom, Triangle, LinearBVHNode, Material, Texture, ...

// Host-only bookkeeping for releasing a texture at teardown. These handles are
// never uploaded to the device, which is why they live here rather than in the
// device-side Texture struct.
struct TextureResource {
    cudaTextureObject_t texObj; // to cudaDestroyTextureObject
    cudaArray_t array;          // to cudaFreeArray (backing storage)
};

// Owns every allocation that lives in device memory for a render, plus the
// host-only handles needed to release them. Created/filled by deviceSceneInit
// and released by deviceSceneFree; the render kernels in pathtrace.cu read the
// pointers off of it. This replaces what used to be ~15 file-scope globals so
// ownership of the GPU resources is explicit and in one place.
struct DeviceScene {
    // Geometry + flattened BVH (one concatenated array each, indexed per geom).
    Geom *geoms = nullptr;
    Geom *lights = nullptr;
    Triangle *geomTriangles = nullptr;
    Triangle *lightTriangles = nullptr;
    LinearBVHNode *geomBVHNodes = nullptr;
    LinearBVHNode *lightBVHNodes = nullptr;
    int *totalNumberOfLights = nullptr;

    // Materials + textures (arrays of hardware texture handles).
    Material *materials = nullptr;
    Texture *albedoTextures = nullptr;
    Texture *normalTextures = nullptr;
    Texture *bumpTextures = nullptr;

    // Equirectangular HDR environment map (valid == 0 when none configured).
    // Backed by a texture resource tracked in `textureResources` for teardown.
    EnvironmentMap envMap = {};

    // Per-render buffers.
    glm::vec3 *image = nullptr;
    PathSegment *paths = nullptr;
    ShadeableIntersection *intersections = nullptr;

    // Host-only: handles for releasing the texture objects + backing arrays.
    std::vector<TextureResource> textureResources;
};

// Allocate and upload everything in `scene` to the device, filling `ds`. Also
// frees the host-side mesh triangle/BVH copies once they're on the device.
void deviceSceneInit(DeviceScene &ds, Scene *scene);

// Release every device allocation and texture resource owned by `ds`.
void deviceSceneFree(DeviceScene &ds);
