#include "bvh.h"
#include "sceneStructs.h"

void BVHNode::initializeLeaf(int first, int n, const glm::vec3 &boxMin,
                             const glm::vec3 &boxMax) {
    firstPrimOffset = first;
    nPrimitives = n;
    bboxMin = boxMin;
    bboxMax = boxMax;

    // Remember to allocate memory to the children array first
    for (int i = 0; i < BVH_CHILDREN; ++i) {
        children[i] = nullptr;
    }
}

void BVHNode::initializeInterior(int axis, BVHNode *c[BVH_CHILDREN]) {
    children[0] = c[0];
    glm::vec3 finalMin(c[0]->bboxMin);
    glm::vec3 finalMax(c[0]->bboxMax);
    for (int i = 1; i < BVH_CHILDREN; ++i) {
        children[i] = c[i];
        finalMin = glm::min(finalMin, c[i]->bboxMin);
        finalMax = glm::max(finalMax, c[i]->bboxMax);
    }
    splitAxis = axis;
    bboxMin = finalMin;
    bboxMax = finalMax;
    nPrimitives = 0;
}

BVH::BVH(vector<Triangle> &triangles) {}

void BVH::initialize(vector<Triangle> &triangles) {
    vector<BVHTriangle> bvhTriangles(triangles.size());
    for (int i = 0; i < triangles.size(); ++i) {
        bvhTriangles[i] = BVHTriangle(i, triangles[i]);
    }

    BVHNode *root;
}

BVHNode *BVH::buildBVH() {
    glm::vec3 bounds[2];
    return nullptr;
}