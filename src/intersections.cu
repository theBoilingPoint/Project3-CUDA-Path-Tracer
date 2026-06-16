#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    // From original code base. But I'd rather handle the normal by myself.
    // if (!outside)
    // {
    //     normal = -normal;
    // }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTestNaive(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside) {

    float t = INFINITY;
    glm::vec3 finalIntersectionPoint;
    glm::vec3 finalNormal;
    glm::vec2 finalUV;  // Store the final UV coordinates
    bool finalOutside;

    // Transform ray to local space
    glm::vec3 originLocal = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 directionLocal = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    for (int i = 0; i < mesh.numTriangles; i++) {
        const Triangle &tri = mesh.devTriangles[i];

        glm::vec2 baryCoords;
        float hitDist;
        bool hit = glm::intersectRayTriangle(
            originLocal, directionLocal,
            tri.points[0], tri.points[1], tri.points[2],
            baryCoords, hitDist);

        if (!hit) {
            // Try reversed winding to hit back faces
            glm::vec2 baryRev;
            hit = glm::intersectRayTriangle(
                originLocal, directionLocal,
                tri.points[0], tri.points[2], tri.points[1],
                baryRev, hitDist);
            if (hit) {
                // Swap coords: baryRev.x=weight(points[2]), baryRev.y=weight(points[1])
                baryCoords = glm::vec2(baryRev.y, baryRev.x);
            }
        }

        if (!hit || hitDist >= t) {
            continue;
        }

        t = hitDist;

        // baryCoords.x = weight for points[1], baryCoords.y = weight for points[2]
        const float u = baryCoords.x;
        const float v = baryCoords.y;
        const float w = 1.0f - u - v;

        glm::vec3 intersectionPointLocal = w * tri.points[0] +
                                           u * tri.points[1] +
                                           v * tri.points[2];

        finalIntersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPointLocal, 1.0f));

        glm::vec3 normalLocal = glm::normalize(w * tri.normals[0] +
                                               u * tri.normals[1] +
                                               v * tri.normals[2]);

        finalNormal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normalLocal, 0.0f)));

        glm::vec2 uvLocal = w * tri.uvs[0] +
                            u * tri.uvs[1] +
                            v * tri.uvs[2];

        finalUV = uvLocal;

        // Determine if the ray is coming from outside the object (negative dot product)
        finalOutside = glm::dot(finalNormal, r.direction) < 0;
    }

    // If no intersection, return -1
    if (t == INFINITY) {
        return -1;
    }

    // Pass back intersection results
    intersectionPoint = finalIntersectionPoint;
    normal = finalNormal;
    uv = glm::clamp(finalUV, 0.0f, 1.0f);
    outside = finalOutside;

    // r.direction should be normalised so we don't need to devided by the length of r.direction
    return glm::distance(r.origin, finalIntersectionPoint);
}
