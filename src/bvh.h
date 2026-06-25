/*
 * We build the BVH on the C++ side then traverse on the GPU.
 * The intuition is that we are building a offline path tracer hence quality >
 * speed. We can build high quality BVH offline.
 */
#pragma once

#include "glm/common.hpp"
#include "glm/glm.hpp"
#include "sceneStructs.h"
#include <algorithm>
#include <span>
#include <vector>

using namespace std;

// Here we use BVH8. Arguably, we can also use BVH4 or a binary tree.
// This class is used for constructing the BVH.
struct BVHNode {
    BoundingBox bbox;
    // only leaf nodes will store actual geometry references
    BVHNode *children[2];
    // which axis was used to split this node
    int splitAxis;
    // offset to the first primitive in the global primitive array
    int firstPrimOffset;
    // number of primitives stored in this node, marking the end (non inclusive)
    // of the list of primitives stored in this node
    int nPrimitives;

    void initializeLeaf(int first, int n, const BoundingBox &bbox);
    void initializeInterior(int axis, BVHNode *c[2]);
};

struct BVHTriangle {
    size_t triangleIndex;
    BoundingBox bbox;

    BVHTriangle() = default;
    BVHTriangle(size_t idx, BoundingBox box) : triangleIndex(idx), bbox(box) {}

    glm::vec3 centroid() const { return 0.5f * bbox.min + 0.5f * bbox.max; }
};

struct BVHSplitBucket {
    int count = 0;
    BoundingBox bbox;
};

class BVH {
  private:
    int maxPrimsInNode;

    void initialize(vector<Triangle> &tris);
    BVHNode *buildBVH(vector<Triangle> &tris, span<BVHTriangle> bvhTriangles,
                      int *totalNodes, int &orderedPrimsOffset,
                      vector<Triangle> &orderedPrims);
    int flattenBVH(BVHNode *node, int *offset);
    void freeBVHTree(BVHNode *node);

  public:
    vector<Triangle> triangles;
    // TODO: Remember to clean up this nodes pointer on the host side once it's
    // copied to the device
    LinearBVHNode *nodes = nullptr;

    BVH(vector<Triangle> &tris, int _maxPrimsInNode = 1);
};
