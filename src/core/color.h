#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Display transform for linear scene-referred radiance. The renderer keeps its
// accumulation buffer linear; this is used only for the interactive preview
// and 8-bit PNG output.
__host__ __device__ inline glm::vec3 acesFitted(glm::vec3 x) {
    x = glm::max(x, glm::vec3(0.0f));
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                      glm::vec3(0.0f), glm::vec3(1.0f));
}

__host__ __device__ inline float linearToSRGB(float x) {
    x = fmaxf(x, 0.0f);
    return x <= 0.0031308f ? 12.92f * x
                           : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

__host__ __device__ inline glm::vec3 displayTransform(glm::vec3 linear,
                                                       float exposure,
                                                       int toneMap) {
    linear *= exp2f(exposure);
    if (!toneMap) {
        // Backward-compatible path for existing scenes.
        return glm::clamp(linear, glm::vec3(0.0f), glm::vec3(1.0f));
    }
    glm::vec3 mapped = acesFitted(linear);
    return glm::clamp(
        glm::vec3(linearToSRGB(mapped.x), linearToSRGB(mapped.y),
                  linearToSRGB(mapped.z)),
        glm::vec3(0.0f), glm::vec3(1.0f));
}
