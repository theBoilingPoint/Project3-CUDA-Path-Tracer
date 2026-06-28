#include "intersections.h"

__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
                                              glm::vec3 &intersectionPoint,
                                              glm::vec3 &normal,
                                              bool &outside) {
    Ray q;
    q.origin =
        multiplyMV(box.transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.transform.inverseTransform,
                                            glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform.transform,
                                       glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(
            multiplyMV(box.transform.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
                                                 glm::vec3 &intersectionPoint,
                                                 glm::vec3 &normal,
                                                 bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.transform.inverseTransform,
                              glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.transform.inverseTransform,
                                             glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection -
                     (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform.transform,
                                   glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.transform.invTranspose,
                                       glm::vec4(objspaceIntersection, 0.f)));
    // From original code base. But I'd rather handle the normal by myself.
    // if (!outside)
    // {
    //     normal = -normal;
    // }

    return glm::length(r.origin - intersectionPoint);
}

// Slab test: does the ray (origin + t*dir, t in [0, tMax]) hit the AABB?
// invDir = 1/dir per component; the sign handling is folded into the min/max
// swap so dirIsNeg is not needed here.
__host__ __device__ inline bool aabbIntersectP(const BoundingBox &b,
                                               const glm::vec3 &origin,
                                               const glm::vec3 &invDir,
                                               float tMax) {
    float t0 = 0.0f;
    float t1 = tMax;
    for (int i = 0; i < 3; ++i) {
        float tNear = (b.min[i] - origin[i]) * invDir[i];
        float tFar = (b.max[i] - origin[i]) * invDir[i];
        if (tNear > tFar) {
            float tmp = tNear;
            tNear = tFar;
            tFar = tmp;
        }
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

// Computes the model-local surface tangent for a triangle from its UV
// gradients. The tangent points along +U in texture space, which is the frame
// tangent-space normal/bump maps are authored against. Falls back to an edge
// when the UVs are degenerate.
__host__ __device__ inline glm::vec3
computeTriangleTangent(const Triangle &tri) {
    glm::vec3 e1 = tri.points[1] - tri.points[0];
    glm::vec3 e2 = tri.points[2] - tri.points[0];
    glm::vec2 duv1 = tri.uvs[1] - tri.uvs[0];
    glm::vec2 duv2 = tri.uvs[2] - tri.uvs[0];
    float det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (fabsf(det) < 1e-8f) {
        return e1;
    }
    return (duv2.y * e1 - duv1.y * e2) / det;
}

__host__ __device__ float
meshIntersectionTestBVH(Geom mesh, Ray r, glm::vec3 &intersectionPoint,
                        glm::vec3 &normal, glm::vec3 &tangent, glm::vec2 &uv,
                        bool &outside) {
    const LinearBVHNode *nodes = mesh.geometry.devNodes;
    const Triangle *tris = mesh.geometry.devTriangles;
    if (nodes == nullptr) {
        return -1;
    }

    // The BVH (node bounds and triangles) lives in the mesh's local space, so
    // we traverse there and only transform the final hit back to world space.
    glm::vec3 originLocal =
        multiplyMV(mesh.transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 directionLocal = glm::normalize(multiplyMV(
        mesh.transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 invDir(1.0f / directionLocal.x, 1.0f / directionLocal.y,
                     1.0f / directionLocal.z);
    int dirIsNeg[3] = {invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f};

    // Closest hit so far, tracked in local-ray distance units (directionLocal
    // is normalized, matching glm::intersectRayTriangle's hitDist).
    float t = INFINITY;
    glm::vec3 finalIntersectionPoint;
    glm::vec3 finalNormal;
    glm::vec3 finalTangent(0.0f);
    glm::vec2 finalUV;
    bool finalOutside = false;
    bool hitAnything = false;

    int toVisitOffset = 0;
    int currentNodeIndex = 0;
    int nodesToVisit[64];

    while (true) {
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        if (aabbIntersectP(node->bbox, originLocal, invDir, t)) {
            if (node->nPrimitives > 0) {
                // Leaf node: test each triangle.
                for (int i = 0; i < node->nPrimitives; ++i) {
                    const Triangle &tri = tris[node->primitivesOffset + i];

                    glm::vec2 baryCoords;
                    float hitDist;
                    bool hit = glm::intersectRayTriangle(
                        originLocal, directionLocal, tri.points[0],
                        tri.points[1], tri.points[2], baryCoords, hitDist);

                    if (!hit) {
                        // Try reversed winding to hit back faces.
                        glm::vec2 baryRev;
                        hit = glm::intersectRayTriangle(
                            originLocal, directionLocal, tri.points[0],
                            tri.points[2], tri.points[1], baryRev, hitDist);
                        if (hit) {
                            baryCoords = glm::vec2(baryRev.y, baryRev.x);
                        }
                    }

                    if (!hit || hitDist >= t) {
                        continue;
                    }

                    t = hitDist;
                    hitAnything = true;

                    const float u = baryCoords.x;
                    const float v = baryCoords.y;
                    const float w = 1.0f - u - v;

                    glm::vec3 intersectionPointLocal = w * tri.points[0] +
                                                       u * tri.points[1] +
                                                       v * tri.points[2];
                    finalIntersectionPoint =
                        multiplyMV(mesh.transform.transform,
                                   glm::vec4(intersectionPointLocal, 1.0f));

                    glm::vec3 normalLocal =
                        glm::normalize(w * tri.normals[0] + u * tri.normals[1] +
                                       v * tri.normals[2]);
                    finalNormal = glm::normalize(
                        multiplyMV(mesh.transform.invTranspose,
                                   glm::vec4(normalLocal, 0.0f)));

                    // Tangents transform with the model matrix (like
                    // positions), not the inverse-transpose used for normals.
                    glm::vec3 tangentLocal = computeTriangleTangent(tri);
                    finalTangent = multiplyMV(mesh.transform.transform,
                                              glm::vec4(tangentLocal, 0.0f));

                    finalUV = w * tri.uvs[0] + u * tri.uvs[1] + v * tri.uvs[2];
                    finalOutside = glm::dot(finalNormal, r.direction) < 0;
                }

                if (toVisitOffset == 0) {
                    break;
                }
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                // Interior node: visit the near child first, stack the far one.
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) {
                break;
            }
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

    if (!hitAnything) {
        return -1;
    }

    intersectionPoint = finalIntersectionPoint;
    normal = finalNormal;
    tangent = finalTangent;
    uv = glm::clamp(finalUV, 0.0f, 1.0f);
    outside = finalOutside;

    return glm::distance(r.origin, finalIntersectionPoint);
}

__host__ __device__ float
meshIntersectionTestNaive(Geom mesh, Ray r, glm::vec3 &intersectionPoint,
                          glm::vec3 &normal, glm::vec3 &tangent, glm::vec2 &uv,
                          bool &outside) {

    float t = INFINITY;
    glm::vec3 finalIntersectionPoint;
    glm::vec3 finalNormal;
    glm::vec3 finalTangent(0.0f);
    glm::vec2 finalUV; // Store the final UV coordinates
    bool finalOutside;

    // Transform ray to local space
    glm::vec3 originLocal =
        multiplyMV(mesh.transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 directionLocal = glm::normalize(multiplyMV(
        mesh.transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    for (int i = 0; i < mesh.geometry.numTriangles; i++) {
        const Triangle &tri = mesh.geometry.devTriangles[i];

        glm::vec2 baryCoords;
        float hitDist;
        bool hit = glm::intersectRayTriangle(
            originLocal, directionLocal, tri.points[0], tri.points[1],
            tri.points[2], baryCoords, hitDist);

        if (!hit) {
            // Try reversed winding to hit back faces
            glm::vec2 baryRev;
            hit = glm::intersectRayTriangle(originLocal, directionLocal,
                                            tri.points[0], tri.points[2],
                                            tri.points[1], baryRev, hitDist);
            if (hit) {
                // Swap coords: baryRev.x=weight(points[2]),
                // baryRev.y=weight(points[1])
                baryCoords = glm::vec2(baryRev.y, baryRev.x);
            }
        }

        if (!hit || hitDist >= t) {
            continue;
        }

        t = hitDist;

        // baryCoords.x = weight for points[1], baryCoords.y = weight for
        // points[2]
        const float u = baryCoords.x;
        const float v = baryCoords.y;
        const float w = 1.0f - u - v;

        glm::vec3 intersectionPointLocal =
            w * tri.points[0] + u * tri.points[1] + v * tri.points[2];

        finalIntersectionPoint = multiplyMV(
            mesh.transform.transform, glm::vec4(intersectionPointLocal, 1.0f));

        glm::vec3 normalLocal = glm::normalize(
            w * tri.normals[0] + u * tri.normals[1] + v * tri.normals[2]);

        finalNormal = glm::normalize(multiplyMV(mesh.transform.invTranspose,
                                                glm::vec4(normalLocal, 0.0f)));

        // Tangents transform with the model matrix (like positions).
        glm::vec3 tangentLocal = computeTriangleTangent(tri);
        finalTangent =
            multiplyMV(mesh.transform.transform, glm::vec4(tangentLocal, 0.0f));

        glm::vec2 uvLocal = w * tri.uvs[0] + u * tri.uvs[1] + v * tri.uvs[2];

        finalUV = uvLocal;

        // Determine if the ray is coming from outside the object (negative dot
        // product)
        finalOutside = glm::dot(finalNormal, r.direction) < 0;
    }

    // If no intersection, return -1
    if (t == INFINITY) {
        return -1;
    }

    // Pass back intersection results
    intersectionPoint = finalIntersectionPoint;
    normal = finalNormal;
    tangent = finalTangent;
    uv = glm::clamp(finalUV, 0.0f, 1.0f);
    outside = finalOutside;

    // r.direction should be normalised so we don't need to devided by the
    // length of r.direction
    return glm::distance(r.origin, finalIntersectionPoint);
}
