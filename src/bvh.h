/*
 * We build the BVH on the C++ side then traverse on the GPU.
 * The intuition is that we are building a offline path tracer hence quality >
 * speed. We can build high quality BVH offline.
 */
#pragma once

#include "glm/glm.hpp"
#include "sceneStructs.h"
#include <vector>

#define BVH_CHILDREN 8

using namespace std;

// Here we use BVH8. Arguably, we can also use BVH4 or a binary tree.
// This class is used for constructing the BVH.
class BVHNode {
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    // only leaf nodes will store actual geometry references
    BVHNode *children[BVH_CHILDREN];
    // which axis was used to split this node
    int splitAxis;
    // offset to the first primitive in the global primitive array
    int firstPrimOffset;
    // number of primitives stored in this node, marking the end (non inclusive)
    // of the list of primitives stored in this node
    int nPrimitives;

    void initializeLeaf(int first, int n, const glm::vec3 &boxMin,
                        const glm::vec3 &boxMax);

    void initializeInterior(int axis, BVHNode *c[BVH_CHILDREN]);
};

class BVH {
    BVH(vector<Triangle> &triangles);

  private:
    void initialize(vector<Triangle> &triangles);
    BVHNode *buildBVH();
};
