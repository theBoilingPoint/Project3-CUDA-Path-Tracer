#include "bvh.h"
#include "sceneStructs.h"
#include <cassert>

void BVHNode::initializeLeaf(int first, int n, const BoundingBox &box) {
    firstPrimOffset = first;
    nPrimitives = n;
    bbox = box;

    // Remember to allocate memory to the children array first
    for (int i = 0; i < 2; ++i) {
        children[i] = nullptr;
    }
}

void BVHNode::initializeInterior(int axis, BVHNode *c[2]) {
    children[0] = c[0];
    glm::vec3 finalMin = c[0]->bbox.min;
    glm::vec3 finalMax = c[0]->bbox.max;
    for (int i = 1; i < 2; ++i) {
        children[i] = c[i];
        finalMin = glm::min(finalMin, c[i]->bbox.min);
        finalMax = glm::max(finalMax, c[i]->bbox.max);
    }
    splitAxis = axis;
    bbox = BoundingBox(finalMin, finalMax);
    nPrimitives = 0;
}

BVH::BVH(vector<Triangle> &tris, int _maxPrimsInNode)
    : maxPrimsInNode(min(255, _maxPrimsInNode)) {
    initialize(tris);
}

BVH::~BVH() { freeLinearBVHTree(); }

void BVH::initialize(vector<Triangle> &tris) {
    vector<BVHTriangle> bvhTriangles(tris.size());
    for (int i = 0; i < tris.size(); ++i) {
        bvhTriangles[i] = BVHTriangle(i, tris[i]);
    }

    int totalNodes = 0;
    int orderedPrimsOffset = 0;
    vector<Triangle> orderedPrims(tris.size());
    BVHNode *root = buildBVH(tris, bvhTriangles, &totalNodes,
                             orderedPrimsOffset, orderedPrims);
    triangles.swap(orderedPrims);

    bvhTriangles.resize(0);
    bvhTriangles.shrink_to_fit();
    nodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    flattenBVH(root, &offset);

    // The temporary pointer-based tree is no longer needed once flattened.
    freeBVHTree(root);
}

BVHNode *BVH::buildBVH(vector<Triangle> &tris, span<BVHTriangle> bvhTriangles,
                       int *totalNodes, int &orderedPrimsOffset,
                       vector<Triangle> &orderedPrims) {
    ++*totalNodes;
    BVHNode *node = new BVHNode();
    BoundingBox bbox;
    for (const auto &tri : bvhTriangles) {
        bbox.min = glm::min(tri.bbox.min, bbox.min);
        bbox.max = glm::max(tri.bbox.max, bbox.max);
    }

    if (bbox.surfaceArea() == 0 || bvhTriangles.size() == 1) {
        // The recursion has bottomed out. Create a leaf node.
        // Sequential equivalent of PBRT's atomic fetch_add: take the current
        // offset, then advance it by the number of primitives in this leaf.
        int firstPrimOffset = orderedPrimsOffset;
        orderedPrimsOffset += bvhTriangles.size();
        for (size_t i = 0; i < bvhTriangles.size(); ++i) {
            int index = bvhTriangles[i].triangleIndex;
            orderedPrims[firstPrimOffset + i] = tris[index];
        }

        node->initializeLeaf(firstPrimOffset, bvhTriangles.size(), bbox);
        return node;
    } else {
        BoundingBox centroidBbox;
        for (const auto &tri : bvhTriangles) {
            centroidBbox.min = glm::min(centroidBbox.min, tri.centroid());
            centroidBbox.max = glm::max(centroidBbox.max, tri.centroid());
        }
        int dim = centroidBbox.maxDimension();

        if (centroidBbox.max[dim] == centroidBbox.min[dim]) {
            // Create a leaf node.
            int firstPrimOffset = orderedPrimsOffset;
            orderedPrimsOffset += bvhTriangles.size();
            for (size_t i = 0; i < bvhTriangles.size(); ++i) {
                int index = bvhTriangles[i].triangleIndex;
                orderedPrims[firstPrimOffset + i] = tris[index];
            }

            node->initializeLeaf(firstPrimOffset, bvhTriangles.size(), bbox);
            return node;
        } else {
            int mid = bvhTriangles.size() / 2;
            // Default splitting using SAH.
            if (bvhTriangles.size() <= 2) {
                mid = bvhTriangles.size() / 2;
                nth_element(bvhTriangles.begin(), bvhTriangles.begin() + mid,
                            bvhTriangles.end(),
                            [dim](const BVHTriangle &a, const BVHTriangle &b) {
                                return a.centroid()[dim] < b.centroid()[dim];
                            });
            } else {
                // An improvement may be to increase this value when there are
                // many primitives and to decrease it when there are few.
                constexpr int nBuckets = 12;
                BVHSplitBucket buckets[nBuckets];

                for (const auto &prim : bvhTriangles) {
                    int b =
                        nBuckets * centroidBbox.offset(prim.centroid())[dim];
                    if (b == nBuckets) {
                        b = nBuckets - 1;
                    }
                    buckets[b].count++;
                    buckets[b].bbox.min =
                        glm::min(buckets[b].bbox.min, prim.bbox.min);
                    buckets[b].bbox.max =
                        glm::max(buckets[b].bbox.max, prim.bbox.max);
                }

                constexpr int nSplits = nBuckets - 1;
                float costs[nSplits] = {};
                int countBelow = 0;
                BoundingBox boundBelow;
                for (int i = 0; i < nSplits; ++i) {
                    boundBelow.min =
                        glm::min(boundBelow.min, buckets[i].bbox.min);
                    boundBelow.max =
                        glm::max(boundBelow.max, buckets[i].bbox.max);
                    countBelow += buckets[i].count;
                    costs[i] += countBelow * boundBelow.surfaceArea();
                }

                int countAbove = 0;
                BoundingBox boundAbove;
                for (int i = nSplits; i >= 1; --i) {
                    boundAbove.min =
                        glm::min(boundAbove.min, buckets[i].bbox.min);
                    boundAbove.max =
                        glm::max(boundAbove.max, buckets[i].bbox.max);
                    countAbove += buckets[i].count;
                    costs[i - 1] += countAbove * boundAbove.surfaceArea();
                }

                int minCostSplitBucket = -1;
                float minCost = numeric_limits<float>::infinity();
                for (int i = 0; i < nSplits; ++i) {
                    // Compute cost for candidate split and update minimum if
                    // necessary
                    if (costs[i] < minCost) {
                        minCost = costs[i];
                        minCostSplitBucket = i;
                    }
                }

                float leafCost = bvhTriangles.size();
                minCost = .5f + minCost / bbox.surfaceArea();
                if (bvhTriangles.size() > maxPrimsInNode ||
                    minCost < leafCost) {
                    auto midIter = partition(
                        bvhTriangles.begin(), bvhTriangles.end(),
                        [=](const BVHTriangle &bp) {
                            int b = nBuckets *
                                    centroidBbox.offset(bp.centroid())[dim];
                            if (b == nBuckets) {
                                b = nBuckets - 1;
                            }
                            return b <= minCostSplitBucket;
                        });
                    mid = midIter - bvhTriangles.begin();
                } else {
                    int firstPrimOffset = orderedPrimsOffset;
                    orderedPrimsOffset += bvhTriangles.size();
                    for (size_t i = 0; i < bvhTriangles.size(); ++i) {
                        int index = bvhTriangles[i].triangleIndex;
                        orderedPrims[firstPrimOffset + i] = tris[index];
                    }
                    node->initializeLeaf(firstPrimOffset, bvhTriangles.size(),
                                         bbox);
                    return node;
                }
            }

            BVHNode *children[2];
            children[0] =
                buildBVH(tris, bvhTriangles.subspan(0, mid), totalNodes,
                         orderedPrimsOffset, orderedPrims);
            children[1] = buildBVH(tris, bvhTriangles.subspan(mid), totalNodes,
                                   orderedPrimsOffset, orderedPrims);

            BVHNode *c[2] = {children[0], children[1]};
            node->initializeInterior(dim, c);
        }
    }

    return node;
}

int BVH::flattenBVH(BVHNode *node, int *offset) {
    LinearBVHNode *linearNode = &nodes[*offset];
    linearNode->bbox = node->bbox;
    int nodeOffset = (*offset)++;

    if (node->nPrimitives > 0) {
        assert(!node->children[0] && !node->children[1]);
        linearNode->primitivesOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
    } else {
        // Create interior flattened BVH node
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;
        flattenBVH(node->children[0], offset);
        linearNode->secondChildOffset = flattenBVH(node->children[1], offset);
    }

    return nodeOffset;
}

// Frees the host-side linear node array. Call this once the array has been
// copied to the device (or when the BVH is no longer needed on the host).
// Safe to call multiple times.
void BVH::freeLinearBVHTree() {
    delete[] nodes;
    nodes = nullptr;
}

// Recursively frees the temporary BVHNode tree built by buildBVH once it has
// been flattened into the linear node array.
void BVH::freeBVHTree(BVHNode *node) {
    if (!node) {
        return;
    }
    freeBVHTree(node->children[0]);
    freeBVHTree(node->children[1]);
    delete node;
}