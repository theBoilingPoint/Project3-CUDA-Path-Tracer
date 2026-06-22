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
class BVHNode {
  private:
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

  public:
    void initializeLeaf(int first, int n, const BoundingBox &bbox);

    void initializeInterior(int axis, BVHNode *c[2]);
};

class BVH {
  private:
    int maxPrimsInNode;

    void initialize(vector<Triangle> &triangles);
    BVHNode *buildBVH(vector<Triangle> &triangles,
                      span<BVHTriangle> bvhTriangles, int *totalNodes,
                      int &orderedPrimsOffset, vector<Triangle> &orderedPrims);

  public:
    BVH(vector<Triangle> &triangles, int _maxPrimsInNode);
};
