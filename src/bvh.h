/*
 * We build the BVH on the C++ side then traverse on the GPU.
 * The intuition is that we are building a offline path tracer hence quality > speed.
 * We can build high quality BVH offline.
 */
#pragma once

#include "glm/glm.hpp"

#define BVH_CHILDREN 8

// Here we use BVH8. Arguably, we can also use BVH4 or a binary tree.
class BVHNode {
	glm::vec3 bboxMin;
	glm::vec3 bboxMax;
	// only leaf nodes will store actual geometry references
	BVHNode* children[BVH_CHILDREN];
	// which axis was used to split this node
	int splitAxis;
	// offset to the first primitive in the global primitive array
	int firstPrimOffset;
	// number of primitives stored in this node, marking the end (non inclusive) of the list of primitives stored in this node
	int nPrimitives;

	void InitLeaf(int first, int n, const glm::vec3& boxMin, const glm::vec3& boxMax) {
		firstPrimOffset = first;
		nPrimitives = n;
		bboxMin = boxMin;
		bboxMax = boxMax;

		// Remember to allocate memory to the children array first
		for (int i = 0; i < BVH_CHILDREN; ++i) {
			children[i] = nullptr;
		}
	}

	void InitInterior(int axis, BVHNode* c[BVH_CHILDREN]) {
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
};

class BVH {

};
