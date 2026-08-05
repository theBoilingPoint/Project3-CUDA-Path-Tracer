#pragma once

#include <cfloat>
#include <glm/glm.hpp>

#include "sceneStructs.h"

struct CombustionFieldSample {
    float density;
    float soot;
    float fuel;
    float reaction;
    float temperature;
    glm::vec3 velocity;
};

struct GridRay {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct GridBrickInterval {
    int pageIndex;
    int brickIndex;
    float tExit;
};

struct GridEmissionPoint {
    bool valid;
    glm::vec3 localPosition;
    CombustionFieldSample fields;
    float pdfLocalVolume;
};

__host__ __device__ inline int volumeLinearIndex(const glm::ivec3 &p,
                                                 const glm::ivec3 &dims) {
    return p.x + dims.x * (p.y + dims.y * p.z);
}

__host__ __device__ inline int brickSampleIndex(int x, int y, int z) {
    return x + kVolumeBrickSampleSize *
                   (y + kVolumeBrickSampleSize * z);
}

__host__ __device__ inline glm::vec4 trilerpVec4(
    const glm::vec4 &c000, const glm::vec4 &c100, const glm::vec4 &c010,
    const glm::vec4 &c110, const glm::vec4 &c001, const glm::vec4 &c101,
    const glm::vec4 &c011, const glm::vec4 &c111, const glm::vec3 &f) {
    const glm::vec4 x00 = glm::mix(c000, c100, f.x);
    const glm::vec4 x10 = glm::mix(c010, c110, f.x);
    const glm::vec4 x01 = glm::mix(c001, c101, f.x);
    const glm::vec4 x11 = glm::mix(c011, c111, f.x);
    return glm::mix(glm::mix(x00, x10, f.y),
                    glm::mix(x01, x11, f.y), f.z);
}

// Trilinearly sample one sparse brick. The positive halo means all eight
// corners live in the same compact allocation even on a brick boundary.
__host__ __device__ inline bool sampleCombustionGridLocal(
    const SparseVolumeGridDevice &grid, const glm::vec3 &pLocal,
    CombustionFieldSample &sample) {
    sample = {};
    if (!grid.valid || grid.pageTable == nullptr || grid.bricks == nullptr) {
        return false;
    }
#if defined(__CUDA_ARCH__)
    const bool useTexture = grid.combustionTexture != 0 &&
                            grid.thermalFlowTexture != 0 &&
                            grid.textureBrickResolution.x > 0 &&
                            grid.textureBrickResolution.y > 0;
    if (!useTexture && (grid.combustionSamples == nullptr ||
                        grid.thermalFlowSamples == nullptr)) {
        return false;
    }
#else
    if (grid.combustionSamples == nullptr ||
        grid.thermalFlowSamples == nullptr) {
        return false;
    }
#endif

    const glm::vec3 extent = grid.localBoundsMax - grid.localBoundsMin;
    glm::vec3 u = (pLocal - grid.localBoundsMin) / extent;
    if (glm::any(glm::lessThan(u, glm::vec3(0.0f))) ||
        glm::any(glm::greaterThan(u, glm::vec3(1.0f)))) {
        return false;
    }

    const glm::vec3 cellP = u * glm::vec3(grid.cellResolution);
    glm::ivec3 cell = glm::ivec3(glm::floor(cellP));
    const glm::ivec3 maxCell = grid.cellResolution - glm::ivec3(1);
    cell = glm::clamp(cell, glm::ivec3(0), maxCell);
    glm::vec3 f = glm::clamp(cellP - glm::vec3(cell), glm::vec3(0.0f),
                             glm::vec3(1.0f));

    const glm::ivec3 brick = cell / grid.brickSize;
    const glm::ivec3 localCell = cell - brick * grid.brickSize;
    const int page = volumeLinearIndex(brick, grid.brickResolution);
    const int brickIndex = grid.pageTable[page];
    if (brickIndex < 0) {
        return false;
    }

    const int x = localCell.x;
    const int y = localCell.y;
    const int z = localCell.z;

    glm::vec4 combustion;
    glm::vec4 thermalFlow;
#if defined(__CUDA_ARCH__)
    if (useTexture) {
        const VolumeBrickMeta &meta = grid.bricks[brickIndex];
        const float tx = meta.textureBaseX + x + f.x + 0.5f;
        const float ty = meta.textureBaseY + y + f.y + 0.5f;
        const float tz = meta.textureBaseZ + z + f.z + 0.5f;
        const float4 c = tex3D<float4>(grid.combustionTexture, tx, ty, tz);
        const float4 t = tex3D<float4>(grid.thermalFlowTexture, tx, ty, tz);
        combustion = glm::vec4(c.x, c.y, c.z, c.w);
        thermalFlow = glm::vec4(t.x, t.y, t.z, t.w);
    } else
#endif
    {
        const VolumeBrickMeta &meta = grid.bricks[brickIndex];
        const int base = static_cast<int>(meta.sampleOffset);
        const glm::vec4 *c = grid.combustionSamples + base;
        const glm::vec4 *t = grid.thermalFlowSamples + base;
        combustion = trilerpVec4(
            c[brickSampleIndex(x, y, z)],
            c[brickSampleIndex(x + 1, y, z)],
            c[brickSampleIndex(x, y + 1, z)],
            c[brickSampleIndex(x + 1, y + 1, z)],
            c[brickSampleIndex(x, y, z + 1)],
            c[brickSampleIndex(x + 1, y, z + 1)],
            c[brickSampleIndex(x, y + 1, z + 1)],
            c[brickSampleIndex(x + 1, y + 1, z + 1)], f);
        thermalFlow = trilerpVec4(
            t[brickSampleIndex(x, y, z)],
            t[brickSampleIndex(x + 1, y, z)],
            t[brickSampleIndex(x, y + 1, z)],
            t[brickSampleIndex(x + 1, y + 1, z)],
            t[brickSampleIndex(x, y, z + 1)],
            t[brickSampleIndex(x + 1, y, z + 1)],
            t[brickSampleIndex(x, y + 1, z + 1)],
            t[brickSampleIndex(x + 1, y + 1, z + 1)], f);
    }

    sample.density = fmaxf(combustion.x, 0.0f);
    sample.soot = fmaxf(combustion.y, 0.0f);
    sample.fuel = fmaxf(combustion.z, 0.0f);
    sample.reaction = fmaxf(combustion.w, 0.0f);
    sample.temperature = fmaxf(thermalFlow.x, 0.0f);
    sample.velocity = glm::vec3(thermalFlow.y, thermalFlow.z, thermalFlow.w);
    return true;
}

__host__ __device__ inline GridRay makeGridRay(const Geom &g,
                                               const Ray &ray) {
    GridRay local;
    local.origin = glm::vec3(
        g.transform.inverseTransform * glm::vec4(ray.origin, 1.0f));
    local.direction = glm::vec3(
        g.transform.inverseTransform * glm::vec4(ray.direction, 0.0f));
    return local;
}

// Locate the brick containing ray(t), returning the next boundary in the
// original world-ray parameter. The probe offset selects the brick on the
// forward side of an exact boundary and prevents zero-length DDA intervals.
__host__ __device__ inline GridBrickInterval locateGridBrick(
    const SparseVolumeGridDevice &grid, const GridRay &ray, float t,
    float tMax) {
    GridBrickInterval result{-1, -1, tMax};
    const float probeEps = fmaxf(2e-5f, fabsf(t) * 2e-6f);
    const float probeT = fminf(t + probeEps, tMax);
    const glm::vec3 p = ray.origin + probeT * ray.direction;
    const glm::vec3 extent = grid.localBoundsMax - grid.localBoundsMin;
    glm::vec3 u = (p - grid.localBoundsMin) / extent;
    u = glm::clamp(u, glm::vec3(0.0f),
                   glm::vec3(1.0f - 1e-7f));
    glm::ivec3 b = glm::ivec3(glm::floor(
        u * glm::vec3(grid.brickResolution)));
    b = glm::clamp(b, glm::ivec3(0),
                   grid.brickResolution - glm::ivec3(1));

    float exitT = tMax;
    for (int axis = 0; axis < 3; ++axis) {
        const float d = ray.direction[axis];
        if (fabsf(d) < 1e-12f) {
            continue;
        }
        const int face = d > 0.0f ? b[axis] + 1 : b[axis];
        const float boundary =
            grid.localBoundsMin[axis] +
            extent[axis] * ((float)face /
                            (float)grid.brickResolution[axis]);
        const float ta = (boundary - ray.origin[axis]) / d;
        if (ta > t + probeEps * 0.25f) {
            exitT = fminf(exitT, ta);
        }
    }
    result.tExit = fminf(fmaxf(exitT, t + probeEps), tMax);
    result.pageIndex = volumeLinearIndex(b, grid.brickResolution);
    result.brickIndex = grid.pageTable[result.pageIndex];
    return result;
}

__host__ __device__ inline Spectrum gridSigmaA(
    const CombustionFieldSample &f, const Spectrum &smokeA,
    const Spectrum &sootA, const Spectrum &flameA) {
    return smokeA * f.density + sootA * f.soot +
           flameA * f.reaction;
}

__host__ __device__ inline Spectrum gridSigmaS(
    const CombustionFieldSample &f, const Spectrum &smokeS,
    const Spectrum &sootS) {
    return smokeS * f.density + sootS * f.soot;
}

__host__ __device__ inline float gridBrickMajorant(
    const VolumeBrickMeta &meta, const Spectrum &smokeA,
    const Spectrum &smokeS, const Spectrum &sootA, const Spectrum &sootS,
    const Spectrum &flameA) {
    const float bound =
        meta.maxDensity * maxComponent(smokeA + smokeS) +
        meta.maxSoot * maxComponent(sootA + sootS) +
        meta.maxReaction * maxComponent(flameA);
    // The host maxima include interpolation halos. This small outward
    // rounding margin additionally covers device fused-operation and spectral
    // uplift roundoff so the null-collision rate remains strictly
    // conservative.
    return bound * 1.02f + 1e-6f;
}

__host__ __device__ inline float gridEmissionSource(
    const CombustionFieldSample &f, float sootEmission) {
    const float incandescent =
        glm::smoothstep(kVolumeSootEmissionOnsetTemperature, 1650.0f,
                        f.temperature);
    return f.reaction + sootEmission * f.soot * incandescent;
}

__host__ __device__ inline int sampleNormalizedCdf(
    const float *cdf, int count, float u) {
    if (cdf == nullptr || count <= 0) {
        return -1;
    }
    int lo = 0;
    int hi = count;
    const float target = glm::clamp(u, 0.0f, 1.0f - 1e-7f);
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (target < cdf[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return glm::clamp(lo, 0, count - 1);
}

// Two-level emitted-power sampling: first an active brick, then one of its
// 512 cells, then a uniform point inside that cell.
__host__ __device__ inline GridEmissionPoint sampleGridEmissionPoint(
    const SparseVolumeGridDevice &grid, float uBrick, float uCell,
    const glm::vec3 &uPoint) {
    GridEmissionPoint out{};
    if (!grid.valid || grid.activeBrickCount <= 0 ||
        grid.totalEmissionPower <= 0.0f ||
        grid.brickEmissionCdf == nullptr ||
        grid.cellEmissionCdf == nullptr) {
        return out;
    }

    const int brickIndex = sampleNormalizedCdf(
        grid.brickEmissionCdf, grid.activeBrickCount, uBrick);
    if (brickIndex < 0) {
        return out;
    }
    const VolumeBrickMeta &meta = grid.bricks[brickIndex];
    const float brickLower =
        brickIndex > 0 ? grid.brickEmissionCdf[brickIndex - 1] : 0.0f;
    const float pBrick =
        grid.brickEmissionCdf[brickIndex] - brickLower;
    if (pBrick <= 0.0f) {
        return out;
    }

    const float *cellCdf =
        grid.cellEmissionCdf + static_cast<int>(meta.cellCdfOffset);
    const int cellIndex =
        sampleNormalizedCdf(cellCdf, kVolumeBrickCellCount, uCell);
    if (cellIndex < 0) {
        return out;
    }
    const float cellLower = cellIndex > 0 ? cellCdf[cellIndex - 1] : 0.0f;
    const float pCell = cellCdf[cellIndex] - cellLower;
    if (pCell <= 0.0f) {
        return out;
    }

    const int cx = cellIndex % kVolumeBrickSize;
    const int cy =
        (cellIndex / kVolumeBrickSize) % kVolumeBrickSize;
    const int cz = cellIndex /
                   (kVolumeBrickSize * kVolumeBrickSize);
    const glm::ivec3 globalCell(
        meta.brickX * grid.brickSize + cx,
        meta.brickY * grid.brickSize + cy,
        meta.brickZ * grid.brickSize + cz);
    const glm::vec3 gridU =
        (glm::vec3(globalCell) +
         glm::clamp(uPoint, glm::vec3(0.0f),
                    glm::vec3(1.0f - 1e-7f))) /
        glm::vec3(grid.cellResolution);
    out.localPosition =
        glm::mix(grid.localBoundsMin, grid.localBoundsMax, gridU);
    if (!sampleCombustionGridLocal(grid, out.localPosition, out.fields)) {
        return GridEmissionPoint{};
    }
    out.pdfLocalVolume = pBrick * pCell / grid.cellVolume;
    out.valid = out.pdfLocalVolume > 0.0f;
    return out;
}

// Evaluate the piecewise-constant density used by
// sampleGridEmissionPoint(). This is needed when another proposal (for
// example, the inverse-square near-field proposal in pathtrace.cu) samples a
// point and the two proposals are combined as a one-sample mixture.
__host__ __device__ inline float gridEmissionPointPdfLocal(
    const SparseVolumeGridDevice &grid, const glm::vec3 &pLocal) {
    if (!grid.valid || grid.activeBrickCount <= 0 ||
        grid.totalEmissionPower <= 0.0f ||
        grid.brickEmissionCdf == nullptr ||
        grid.cellEmissionCdf == nullptr || grid.pageTable == nullptr ||
        grid.bricks == nullptr || grid.cellVolume <= 0.0f) {
        return 0.0f;
    }

    const glm::vec3 extent = grid.localBoundsMax - grid.localBoundsMin;
    const glm::vec3 u = (pLocal - grid.localBoundsMin) / extent;
    if (glm::any(glm::lessThan(u, glm::vec3(0.0f))) ||
        glm::any(glm::greaterThan(u, glm::vec3(1.0f)))) {
        return 0.0f;
    }

    glm::ivec3 cell =
        glm::ivec3(glm::floor(u * glm::vec3(grid.cellResolution)));
    cell = glm::clamp(cell, glm::ivec3(0),
                      grid.cellResolution - glm::ivec3(1));
    const glm::ivec3 brick = cell / grid.brickSize;
    const int page = volumeLinearIndex(brick, grid.brickResolution);
    const int brickIndex = grid.pageTable[page];
    if (brickIndex < 0 || brickIndex >= grid.activeBrickCount) {
        return 0.0f;
    }

    const VolumeBrickMeta &meta = grid.bricks[brickIndex];
    const float brickLower =
        brickIndex > 0 ? grid.brickEmissionCdf[brickIndex - 1] : 0.0f;
    const float pBrick =
        grid.brickEmissionCdf[brickIndex] - brickLower;
    if (pBrick <= 0.0f) {
        return 0.0f;
    }

    const glm::ivec3 localCell = cell - brick * grid.brickSize;
    const int cellIndex =
        localCell.x +
        grid.brickSize *
            (localCell.y + grid.brickSize * localCell.z);
    const float *cellCdf =
        grid.cellEmissionCdf + static_cast<int>(meta.cellCdfOffset);
    const float cellLower =
        cellIndex > 0 ? cellCdf[cellIndex - 1] : 0.0f;
    const float pCell = cellCdf[cellIndex] - cellLower;
    return pCell > 0.0f ? pBrick * pCell / grid.cellVolume : 0.0f;
}

__device__ inline void addVolumeStat(unsigned long long *counter,
                                     unsigned long long value) {
    if (counter != nullptr && value != 0) {
        atomicAdd(counter, value);
    }
}

__device__ inline void recordVolumeTrackingStats(
    VolumeTrackingStats *stats, unsigned long long brickVisits,
    unsigned long long emptySkips, unsigned long long nullCollisions,
    unsigned long long realCollisions,
    unsigned long long majorantViolations,
    unsigned long long trackingOverflows) {
    if (stats == nullptr) {
        return;
    }
    addVolumeStat(&stats->brickVisits, brickVisits);
    addVolumeStat(&stats->emptyBrickSkips, emptySkips);
    addVolumeStat(&stats->nullCollisions, nullCollisions);
    addVolumeStat(&stats->realCollisions, realCollisions);
    addVolumeStat(&stats->majorantViolations, majorantViolations);
    addVolumeStat(&stats->trackingOverflows, trackingOverflows);
}
