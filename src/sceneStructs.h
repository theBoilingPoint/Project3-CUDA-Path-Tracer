#pragma once

#include "glm/fwd.hpp"
#include "glm/glm.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType { SPHERE, CUBE, MESH };

enum MatType { DIFFUSE, MIRROR, DIELECTRIC, MICROFACET, TEXTURE };

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

struct Geom {
    enum GeomType type;

    struct {
        int materialid;
        int albedoTextureID;
        int normalTextureID;
        int bumpTextureID;
    } material;
    int numTriangles = 0;

    Triangle *triangles;    // Host-side pointer
    Triangle *devTriangles; // Device-side pointer

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material {
    int type;
    glm::vec3 color;
    glm::vec3 specularColor;
    float roughness;
    float emittance;
    float indexOfRefraction;
};

/****** For Texture Loading ******/
struct Texture {
    glm::ivec2 size;
    glm::vec4 *dev_data;
};

struct TextureValues {
    glm::vec4 albedo;
    glm::vec4 normal;
    glm::vec4 bump;
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
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    bool hasHitLight;
    float eta; // Used for Russian roulette to determine how likely this ray
               // survives
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
    struct {
        int materialId;
        int albedoTextureID;
        int normalTextureID;
        int bumpTextureID;
    } materials;
};
