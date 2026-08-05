#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "render/volume.h"
#include "render/volumeGridDevice.h"
#include "scene/volumeGrid.h"

namespace {

[[noreturn]] void fail(const char *expression, const char *file, int line,
                       const std::string &detail = {}) {
    std::ostringstream message;
    message << file << ':' << line << ": check failed: " << expression;
    if (!detail.empty()) {
        message << " (" << detail << ')';
    }
    throw std::runtime_error(message.str());
}

void requireImpl(bool condition, const char *expression, const char *file,
                 int line, const std::string &detail = {}) {
    if (!condition) {
        fail(expression, file, line, detail);
    }
}

#define REQUIRE(expression)                                                   \
    requireImpl(static_cast<bool>(expression), #expression, __FILE__,         \
                __LINE__)

#define REQUIRE_DETAIL(expression, detail)                                    \
    requireImpl(static_cast<bool>(expression), #expression, __FILE__,         \
                __LINE__, detail)

void requireNear(float actual, float expected, float absoluteTolerance,
                 float relativeTolerance, const char *label) {
    const float scale =
        std::max(std::abs(actual), std::abs(expected));
    const float tolerance =
        absoluteTolerance + relativeTolerance * scale;
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        std::ostringstream detail;
        detail << label << ": actual=" << std::setprecision(9) << actual
               << ", expected=" << expected << ", tolerance=" << tolerance;
        fail("values are approximately equal", __FILE__, __LINE__,
             detail.str());
    }
}

#if SPECTRAL
constexpr int kSpectrumComponentCount = NSpectrumSamples;

Spectrum makeSpectrum(float x, float y, float z, float w) {
    return Spectrum(x, y, z, w);
}
#else
constexpr int kSpectrumComponentCount = 3;

Spectrum makeSpectrum(float x, float y, float z, float) {
    return Spectrum(x, y, z);
}
#endif

void requireSpectrumNear(const Spectrum &actual, const Spectrum &expected,
                         float absoluteTolerance, float relativeTolerance,
                         const char *label) {
    for (int component = 0; component < kSpectrumComponentCount;
         ++component) {
        std::ostringstream componentLabel;
        componentLabel << label << '[' << component << ']';
        requireNear(actual[component], expected[component],
                    absoluteTolerance, relativeTolerance,
                    componentLabel.str().c_str());
    }
}

int presetIndex(CombustionPreset preset) {
    return static_cast<int>(preset);
}

const char *presetName(CombustionPreset preset) {
    switch (preset) {
    case CombustionPreset::Candle:
        return "candle";
    case CombustionPreset::WoodFire:
        return "wood fire";
    case CombustionPreset::Wildfire:
        return "wildfire";
    }
    return "unknown";
}

glm::ivec3 testResolution(CombustionPreset preset) {
    switch (preset) {
    case CombustionPreset::Candle:
        return glm::ivec3(16, 24, 16);
    case CombustionPreset::WoodFire:
        return glm::ivec3(24, 24, 16);
    case CombustionPreset::Wildfire:
        return glm::ivec3(32, 24, 24);
    }
    return glm::ivec3(16);
}

const HostSparseVolumeGrid &smallPresetGrid(CombustionPreset preset) {
    static std::array<std::unique_ptr<HostSparseVolumeGrid>, 3> cache;
    const int index = presetIndex(preset);
    REQUIRE(index >= 0 && index < static_cast<int>(cache.size()));
    if (!cache[static_cast<std::size_t>(index)]) {
        VolumeGridBuildSettings settings =
            defaultVolumeGridBuildSettings(preset);
        settings.cellResolution = testResolution(preset);
        settings.seed = 1031 + 97 * index;
        cache[static_cast<std::size_t>(index)] =
            std::make_unique<HostSparseVolumeGrid>(
                buildSparseCombustionGrid(settings));
    }
    return *cache[static_cast<std::size_t>(index)];
}

SparseVolumeGridDevice deviceView(const HostSparseVolumeGrid &host) {
    SparseVolumeGridDevice device{};
    device.valid = 1;
    device.cellResolution = host.cellResolution;
    device.brickResolution = host.brickResolution;
    device.activeBrickCount = static_cast<int>(host.bricks.size());
    device.localBoundsMin = host.localBoundsMin;
    device.localBoundsMax = host.localBoundsMax;
    device.brickSize = host.brickSize;
    device.cellVolume = host.cellVolume;
    device.totalEmissionPower = host.totalEmissionPower;
    const glm::vec3 extent = host.localBoundsMax - host.localBoundsMin;
    device.localToWorldVolume = extent.x * extent.y * extent.z;
    device.pageTable = host.pageTable.data();
    device.bricks = host.bricks.data();
    device.combustionSamples = host.combustionSamples.data();
    device.thermalFlowSamples = host.thermalFlowSamples.data();
    device.brickEmissionCdf = host.brickEmissionCdf.data();
    device.cellEmissionCdf = host.cellEmissionCdf.data();
    return device;
}

HostSparseVolumeGrid makeLinearRampGrid() {
    HostSparseVolumeGrid grid;
    grid.cellResolution = glm::ivec3(kVolumeBrickSize);
    grid.brickResolution = glm::ivec3(1);
    grid.localBoundsMin = glm::vec3(-2.0f, -1.0f, 0.0f);
    grid.localBoundsMax = glm::vec3(2.0f, 3.0f, 4.0f);
    grid.brickSize = kVolumeBrickSize;
    const glm::vec3 cellSize =
        (grid.localBoundsMax - grid.localBoundsMin) /
        glm::vec3(grid.cellResolution);
    grid.cellVolume = cellSize.x * cellSize.y * cellSize.z;
    grid.totalEmissionPower = 0.0f;
    grid.pageTable = {0};

    VolumeBrickMeta meta;
    meta.brickX = 0;
    meta.brickY = 0;
    meta.brickZ = 0;
    meta.sampleOffset = 0;
    meta.cellCdfOffset = 0;

    grid.combustionSamples.resize(kVolumeBrickSampleCount);
    grid.thermalFlowSamples.resize(kVolumeBrickSampleCount);
    for (int z = 0; z < kVolumeBrickSampleSize; ++z) {
        for (int y = 0; y < kVolumeBrickSampleSize; ++y) {
            for (int x = 0; x < kVolumeBrickSampleSize; ++x) {
                const glm::vec3 q =
                    glm::vec3(x, y, z) /
                    static_cast<float>(kVolumeBrickSize);
                const float density =
                    0.25f + 0.20f * q.x + 0.30f * q.y + 0.10f * q.z;
                const float soot =
                    0.05f + 0.10f * q.x + 0.08f * q.y + 0.07f * q.z;
                const float fuel =
                    0.10f + 0.12f * q.x + 0.04f * q.y + 0.06f * q.z;
                const float reaction =
                    0.02f + 0.03f * q.x + 0.05f * q.y + 0.07f * q.z;
                const float temperature =
                    400.0f + 700.0f * q.x + 600.0f * q.y +
                    500.0f * q.z;
                const glm::vec3 velocity(
                    -0.5f + q.x + 0.2f * q.y,
                    0.1f + 0.3f * q.y + 0.4f * q.z,
                    -0.2f + 0.5f * q.x - 0.1f * q.z);
                const int index = brickSampleIndex(x, y, z);
                grid.combustionSamples[static_cast<std::size_t>(index)] =
                    glm::vec4(density, soot, fuel, reaction);
                grid.thermalFlowSamples[static_cast<std::size_t>(index)] =
                    glm::vec4(temperature, velocity);

                meta.maxDensity = std::max(meta.maxDensity, density);
                meta.maxSoot = std::max(meta.maxSoot, soot);
                meta.maxFuel = std::max(meta.maxFuel, fuel);
                meta.maxReaction = std::max(meta.maxReaction, reaction);
                meta.maxTemperature =
                    std::max(meta.maxTemperature, temperature);
                meta.maxFieldExtinction =
                    std::max(meta.maxFieldExtinction, density + soot);
            }
        }
    }
    grid.bricks.push_back(meta);

    grid.brickEmissionCdf = {1.0f};
    grid.cellEmissionCdf.reserve(kVolumeBrickCellCount);
    for (int cell = 0; cell < kVolumeBrickCellCount; ++cell) {
        grid.cellEmissionCdf.push_back(
            static_cast<float>(cell + 1) /
            static_cast<float>(kVolumeBrickCellCount));
    }
    return grid;
}

CombustionFieldSample expectedLinearRamp(const glm::vec3 &q) {
    CombustionFieldSample sample{};
    sample.density =
        0.25f + 0.20f * q.x + 0.30f * q.y + 0.10f * q.z;
    sample.soot = 0.05f + 0.10f * q.x + 0.08f * q.y + 0.07f * q.z;
    sample.fuel = 0.10f + 0.12f * q.x + 0.04f * q.y + 0.06f * q.z;
    sample.reaction =
        0.02f + 0.03f * q.x + 0.05f * q.y + 0.07f * q.z;
    sample.temperature =
        400.0f + 700.0f * q.x + 600.0f * q.y + 500.0f * q.z;
    sample.velocity =
        glm::vec3(-0.5f + q.x + 0.2f * q.y,
                  0.1f + 0.3f * q.y + 0.4f * q.z,
                  -0.2f + 0.5f * q.x - 0.1f * q.z);
    return sample;
}

void requireSampleNear(const CombustionFieldSample &actual,
                       const CombustionFieldSample &expected,
                       float tolerance) {
    requireNear(actual.density, expected.density, tolerance, tolerance,
                "density");
    requireNear(actual.soot, expected.soot, tolerance, tolerance, "soot");
    requireNear(actual.fuel, expected.fuel, tolerance, tolerance, "fuel");
    requireNear(actual.reaction, expected.reaction, tolerance, tolerance,
                "reaction");
    requireNear(actual.temperature, expected.temperature,
                tolerance * 100.0f, tolerance, "temperature");
    for (int axis = 0; axis < 3; ++axis) {
        requireNear(actual.velocity[axis], expected.velocity[axis],
                    tolerance, tolerance, "velocity");
    }
}

void testPresetBuildAndValidation() {
    const std::array<CombustionPreset, 3> presets = {
        CombustionPreset::Candle, CombustionPreset::WoodFire,
        CombustionPreset::Wildfire};
    for (CombustionPreset preset : presets) {
        const HostSparseVolumeGrid &grid = smallPresetGrid(preset);
        std::string error;
        REQUIRE_DETAIL(validateSparseCombustionGrid(grid, &error),
                       std::string(presetName(preset)) + ": " + error);
        REQUIRE(!grid.bricks.empty());
        REQUIRE(grid.bricks.size() <= grid.pageTable.size());
        REQUIRE(grid.totalEmissionPower >= 0.0f);
        const bool hasCombustionField =
            std::any_of(grid.bricks.begin(), grid.bricks.end(),
                        [](const VolumeBrickMeta &brick) {
                            return brick.maxDensity > 0.0f ||
                                   brick.maxSoot > 0.0f ||
                                   brick.maxFuel > 0.0f ||
                                   brick.maxReaction > 0.0f;
                        });
        REQUIRE_DETAIL(hasCombustionField, presetName(preset));
        REQUIRE(grid.brickEmissionCdf.size() == grid.bricks.size());
        REQUIRE(grid.cellEmissionCdf.size() ==
                grid.bricks.size() * kVolumeBrickCellCount);
    }
}

void testExactSparseTrilinearInterpolation() {
    const HostSparseVolumeGrid host = makeLinearRampGrid();
    std::string error;
    REQUIRE_DETAIL(validateSparseCombustionGrid(host, &error), error);
    const SparseVolumeGridDevice grid = deviceView(host);

    std::mt19937 generator(0x51a7c0deu);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int sampleIndex = 0; sampleIndex < 256; ++sampleIndex) {
        const glm::vec3 q(uniform(generator), uniform(generator),
                          uniform(generator));
        const glm::vec3 p =
            glm::mix(host.localBoundsMin, host.localBoundsMax, q);
        CombustionFieldSample actual{};
        REQUIRE(sampleCombustionGridLocal(grid, p, actual));
        requireSampleNear(actual, expectedLinearRamp(q), 3.0e-6f);
    }

    for (const glm::vec3 q :
         {glm::vec3(0.0f), glm::vec3(1.0f),
          glm::vec3(0.0f, 1.0f, 0.5f),
          glm::vec3(1.0f, 0.0f, 0.5f)}) {
        CombustionFieldSample actual{};
        const glm::vec3 p =
            glm::mix(host.localBoundsMin, host.localBoundsMax, q);
        REQUIRE(sampleCombustionGridLocal(grid, p, actual));
        requireSampleNear(actual, expectedLinearRamp(q), 3.0e-6f);
    }

    CombustionFieldSample outside{};
    REQUIRE(!sampleCombustionGridLocal(
        grid, host.localBoundsMin - glm::vec3(1.0e-3f, 0.0f, 0.0f),
        outside));
}

void testConservativeHaloMajorants() {
    const Spectrum smokeA = makeSpectrum(0.12f, 0.16f, 0.21f, 0.27f);
    const Spectrum smokeS = makeSpectrum(0.26f, 0.22f, 0.18f, 0.14f);
    const Spectrum sootA = makeSpectrum(0.44f, 0.51f, 0.63f, 0.72f);
    const Spectrum sootS = makeSpectrum(0.08f, 0.07f, 0.05f, 0.03f);
    const Spectrum flameA = makeSpectrum(0.03f, 0.04f, 0.06f, 0.08f);
    std::mt19937 generator(0xc0111deu);
    std::uniform_real_distribution<float> uniform(
        0.0f, std::nextafter(1.0f, 0.0f));

    const std::array<CombustionPreset, 3> presets = {
        CombustionPreset::Candle, CombustionPreset::WoodFire,
        CombustionPreset::Wildfire};
    for (CombustionPreset preset : presets) {
        const HostSparseVolumeGrid &host = smallPresetGrid(preset);
        const SparseVolumeGridDevice grid = deviceView(host);
        for (std::size_t brickIndex = 0;
             brickIndex < host.bricks.size(); ++brickIndex) {
            const VolumeBrickMeta &meta = host.bricks[brickIndex];
            const glm::vec3 brickU0 =
                glm::vec3(meta.brickX, meta.brickY, meta.brickZ) /
                glm::vec3(host.brickResolution);
            const glm::vec3 brickU1 =
                glm::vec3(meta.brickX + 1, meta.brickY + 1,
                          meta.brickZ + 1) /
                glm::vec3(host.brickResolution);
            const float majorant =
                gridBrickMajorant(meta, smokeA, smokeS, sootA, sootS,
                                  flameA);
            REQUIRE(majorant >= 0.0f);

            for (int trial = 0; trial < 24; ++trial) {
                const glm::vec3 withinBrick(
                    uniform(generator), uniform(generator),
                    uniform(generator));
                const glm::vec3 q =
                    glm::mix(brickU0, brickU1, withinBrick);
                const glm::vec3 p =
                    glm::mix(host.localBoundsMin, host.localBoundsMax, q);
                CombustionFieldSample fields{};
                REQUIRE(sampleCombustionGridLocal(grid, p, fields));
                const float fieldTolerance = 3.0e-6f;
                REQUIRE(fields.density <=
                        meta.maxDensity + fieldTolerance);
                REQUIRE(fields.soot <= meta.maxSoot + fieldTolerance);
                REQUIRE(fields.fuel <= meta.maxFuel + fieldTolerance);
                REQUIRE(fields.reaction <=
                        meta.maxReaction + fieldTolerance);
                REQUIRE(fields.temperature <=
                        meta.maxTemperature + 2.0e-3f);
                REQUIRE(fields.density + fields.soot <=
                        meta.maxFieldExtinction + 2.0f * fieldTolerance);

                const Spectrum sigmaT =
                    gridSigmaA(fields, smokeA, sootA, flameA) +
                    gridSigmaS(fields, smokeS, sootS);
                REQUIRE(maxComponent(sigmaT) <=
                        majorant * (1.0f + 1.0e-5f) + 2.0e-6f);
            }
        }
    }
}

void requireNormalizedCdf(const float *cdf, std::size_t count,
                          const char *label) {
    REQUIRE(count > 0);
    float previous = 0.0f;
    float probabilitySum = 0.0f;
    for (std::size_t index = 0; index < count; ++index) {
        const float value = cdf[index];
        REQUIRE_DETAIL(std::isfinite(value), label);
        REQUIRE_DETAIL(value + 1.0e-7f >= previous, label);
        REQUIRE_DETAIL(value >= -1.0e-7f && value <= 1.0f + 1.0e-6f,
                       label);
        probabilitySum += value - previous;
        previous = value;
    }
    requireNear(previous, 1.0f, 2.0e-6f, 2.0e-6f, label);
    requireNear(probabilitySum, 1.0f, 2.0e-6f, 2.0e-6f, label);
}

void testPageTableAndEmissionCdfs() {
    const HostSparseVolumeGrid &host =
        smallPresetGrid(CombustionPreset::WoodFire);
    const SparseVolumeGridDevice grid = deviceView(host);

    std::vector<int> references(host.bricks.size(), 0);
    for (std::size_t page = 0; page < host.pageTable.size(); ++page) {
        const int brick = host.pageTable[page];
        REQUIRE(brick >= -1 &&
                brick < static_cast<int>(host.bricks.size()));
        if (brick >= 0) {
            ++references[static_cast<std::size_t>(brick)];
        }
    }
    for (int count : references) {
        REQUIRE(count == 1);
    }

    requireNormalizedCdf(host.brickEmissionCdf.data(),
                         host.brickEmissionCdf.size(),
                         "brick emission CDF");
    float previousBrickCdf = 0.0f;
    for (std::size_t brickIndex = 0;
         brickIndex < host.bricks.size(); ++brickIndex) {
        const VolumeBrickMeta &meta = host.bricks[brickIndex];
        const float pBrick =
            host.brickEmissionCdf[brickIndex] - previousBrickCdf;
        previousBrickCdf = host.brickEmissionCdf[brickIndex];
        const float expected =
            meta.emissionPower / host.totalEmissionPower;
        requireNear(pBrick, expected, 3.0e-7f, 3.0e-5f,
                    "brick emitted-power probability");
        requireNormalizedCdf(
            host.cellEmissionCdf.data() + meta.cellCdfOffset,
            kVolumeBrickCellCount, "cell emission CDF");
    }

    int chosenBrick = -1;
    int chosenCell = -1;
    float brickLower = 0.0f;
    float cellLower = 0.0f;
    float pBrick = 0.0f;
    float pCell = 0.0f;
    for (std::size_t brickIndex = 0;
         brickIndex < host.bricks.size() && chosenBrick < 0;
         ++brickIndex) {
        const float lower =
            brickIndex == 0 ? 0.0f
                            : host.brickEmissionCdf[brickIndex - 1];
        const float probability =
            host.brickEmissionCdf[brickIndex] - lower;
        if (probability <= 1.0e-7f) {
            continue;
        }
        const VolumeBrickMeta &meta = host.bricks[brickIndex];
        const float *cellCdf =
            host.cellEmissionCdf.data() + meta.cellCdfOffset;
        float lowerCell = 0.0f;
        for (int cell = 0; cell < kVolumeBrickCellCount; ++cell) {
            const float probabilityCell = cellCdf[cell] - lowerCell;
            if (probabilityCell > 1.0e-7f) {
                chosenBrick = static_cast<int>(brickIndex);
                chosenCell = cell;
                brickLower = lower;
                cellLower = lowerCell;
                pBrick = probability;
                pCell = probabilityCell;
                break;
            }
            lowerCell = cellCdf[cell];
        }
    }
    REQUIRE(chosenBrick >= 0);
    REQUIRE(chosenCell >= 0);

    const float uBrick = brickLower + 0.5f * pBrick;
    const float uCell = cellLower + 0.5f * pCell;
    REQUIRE(sampleNormalizedCdf(host.brickEmissionCdf.data(),
                                static_cast<int>(
                                    host.brickEmissionCdf.size()),
                                uBrick) == chosenBrick);
    const VolumeBrickMeta &meta =
        host.bricks[static_cast<std::size_t>(chosenBrick)];
    REQUIRE(sampleNormalizedCdf(
                host.cellEmissionCdf.data() + meta.cellCdfOffset,
                kVolumeBrickCellCount, uCell) == chosenCell);

    const glm::vec3 uPoint(0.31f, 0.47f, 0.69f);
    const GridEmissionPoint point =
        sampleGridEmissionPoint(grid, uBrick, uCell, uPoint);
    REQUIRE(point.valid);
    requireNear(point.pdfLocalVolume,
                pBrick * pCell / host.cellVolume, 2.0e-5f, 5.0e-5f,
                "sampled emission volume pdf");
    requireNear(gridEmissionPointPdfLocal(grid, point.localPosition),
                point.pdfLocalVolume, 2.0e-5f, 5.0e-5f,
                "evaluated emission volume pdf");
    REQUIRE(gridEmissionPointPdfLocal(
                grid, host.localBoundsMax + glm::vec3(0.1f)) == 0.0f);
    float evaluatedPdfIntegral = 0.0f;
    for (const VolumeBrickMeta &brick : host.bricks) {
        for (int localZ = 0; localZ < kVolumeBrickSize; ++localZ) {
            for (int localY = 0; localY < kVolumeBrickSize; ++localY) {
                for (int localX = 0; localX < kVolumeBrickSize; ++localX) {
                    const glm::ivec3 globalCell(
                        brick.brickX * kVolumeBrickSize + localX,
                        brick.brickY * kVolumeBrickSize + localY,
                        brick.brickZ * kVolumeBrickSize + localZ);
                    const glm::vec3 centerU =
                        (glm::vec3(globalCell) + glm::vec3(0.5f)) /
                        glm::vec3(host.cellResolution);
                    const glm::vec3 center =
                        glm::mix(host.localBoundsMin, host.localBoundsMax,
                                 centerU);
                    evaluatedPdfIntegral +=
                        gridEmissionPointPdfLocal(grid, center) *
                        host.cellVolume;
                }
            }
        }
    }
    requireNear(evaluatedPdfIntegral, 1.0f, 2.0e-4f, 2.0e-4f,
                "evaluated emission pdf normalization");

    const int cellX = chosenCell % kVolumeBrickSize;
    const int cellY =
        (chosenCell / kVolumeBrickSize) % kVolumeBrickSize;
    const int cellZ =
        chosenCell / (kVolumeBrickSize * kVolumeBrickSize);
    const glm::ivec3 globalCell(
        meta.brickX * kVolumeBrickSize + cellX,
        meta.brickY * kVolumeBrickSize + cellY,
        meta.brickZ * kVolumeBrickSize + cellZ);
    const glm::vec3 expectedU =
        (glm::vec3(globalCell) + uPoint) /
        glm::vec3(host.cellResolution);
    const glm::vec3 expectedPosition =
        glm::mix(host.localBoundsMin, host.localBoundsMax, expectedU);
    for (int axis = 0; axis < 3; ++axis) {
        requireNear(point.localPosition[axis], expectedPosition[axis],
                    2.0e-6f, 2.0e-6f,
                    "emission sample local position");
    }
}

void testEmissionCdfHasCompleteSupport() {
    const std::array<CombustionPreset, 3> presets{
        CombustionPreset::Candle,
        CombustionPreset::WoodFire,
        CombustionPreset::Wildfire,
    };

    for (CombustionPreset preset : presets) {
        const HostSparseVolumeGrid &host = smallPresetGrid(preset);
        float previousBrickCdf = 0.0f;
        for (std::size_t brickIndex = 0;
             brickIndex < host.bricks.size(); ++brickIndex) {
            const VolumeBrickMeta &meta = host.bricks[brickIndex];
            const float pBrick =
                host.brickEmissionCdf[brickIndex] - previousBrickCdf;
            previousBrickCdf = host.brickEmissionCdf[brickIndex];

            const float *cellCdf =
                host.cellEmissionCdf.data() + meta.cellCdfOffset;
            float previousCellCdf = 0.0f;
            for (int cellZ = 0; cellZ < kVolumeBrickSize; ++cellZ) {
                for (int cellY = 0; cellY < kVolumeBrickSize; ++cellY) {
                    for (int cellX = 0; cellX < kVolumeBrickSize;
                         ++cellX) {
                        float maxReaction = 0.0f;
                        float maxSoot = 0.0f;
                        float maxTemperature = 0.0f;
                        for (int dz = 0; dz <= 1; ++dz) {
                            for (int dy = 0; dy <= 1; ++dy) {
                                for (int dx = 0; dx <= 1; ++dx) {
                                    const std::size_t sampleIndex =
                                        static_cast<std::size_t>(
                                            meta.sampleOffset) +
                                        static_cast<std::size_t>(
                                            brickSampleIndex(
                                                cellX + dx, cellY + dy,
                                                cellZ + dz));
                                    const glm::vec4 combustion =
                                        host.combustionSamples[sampleIndex];
                                    const glm::vec4 thermal =
                                        host.thermalFlowSamples[sampleIndex];
                                    maxReaction =
                                        std::max(maxReaction,
                                                 combustion.w);
                                    maxSoot =
                                        std::max(maxSoot, combustion.y);
                                    maxTemperature =
                                        std::max(maxTemperature,
                                                 thermal.x);
                                }
                            }
                        }

                        // With nonnegative trilinear fields, any positive
                        // reaction corner gives reaction support throughout
                        // some part of the cell. Likewise, nonzero soot and a
                        // temperature corner above the gate overlap in the
                        // cell interior even when they occur at different
                        // corners.
                        const bool canEmit =
                            maxReaction > 0.0f ||
                            (maxSoot > 0.0f &&
                             maxTemperature >
                                 kVolumeSootEmissionOnsetTemperature);
                        const int cellIndex =
                            cellX +
                            kVolumeBrickSize *
                                (cellY + kVolumeBrickSize * cellZ);
                        const float cellUpper = cellCdf[cellIndex];
                        const float pCell =
                            cellUpper - previousCellCdf;
                        previousCellCdf = cellUpper;
                        if (!canEmit) {
                            continue;
                        }

                        std::ostringstream detail;
                        detail << presetName(preset) << " brick "
                               << brickIndex << " cell " << cellIndex;
                        REQUIRE_DETAIL(pBrick > 0.0f, detail.str());
                        REQUIRE_DETAIL(pCell > 0.0f, detail.str());
                        REQUIRE_DETAIL(
                            pBrick * pCell / host.cellVolume > 0.0f,
                            detail.str());
                    }
                }
            }
        }
    }
}

void testHomogeneousAnalyticTransmittance() {
    Geom geom{};
    Material material{};
    material.heterogeneous = 0;
    Ray ray{};
    ray.direction = glm::vec3(0.0f, 1.0f, 0.0f);
    thrust::default_random_engine rng(42);
    const Spectrum sigmaT =
        makeSpectrum(0.07f, 0.21f, 0.43f, 0.89f);
    const float t0 = 1.25f;
    const float t1 = 4.75f;
    const Spectrum actual =
        mediumTransmittance(geom, material, sigmaT, ray, t0, t1, rng);
    Spectrum expected(0.0f);
    for (int component = 0; component < kSpectrumComponentCount;
         ++component) {
        expected[component] =
            std::exp(-sigmaT[component] * (t1 - t0));
    }
    requireSpectrumNear(actual, expected, 2.0e-7f, 2.0e-6f,
                        "homogeneous Beer-Lambert transmittance");

    const Spectrum zeroDistance =
        mediumTransmittance(geom, material, sigmaT, ray, t0, t0, rng);
    requireSpectrumNear(zeroDistance, Spectrum(1.0f), 1.0e-7f, 1.0e-7f,
                        "zero-distance transmittance");
}

void testHomogeneousPureAbsorptionFreeFlight() {
    const Spectrum sigmaT =
        makeSpectrum(0.80f, 0.65f, 0.50f, 0.35f);
    const Spectrum sigmaS(0.0f);
    const float distance = 2.0f;
    Spectrum expected(0.0f);
    for (int component = 0; component < kSpectrumComponentCount;
         ++component) {
        expected[component] =
            std::exp(-sigmaT[component] * distance);
    }

    // The pure-absorption branch must not depend on a free-flight random
    // variate: no physical scattering event can occur.
    for (float u : {0.0f, 0.01f, 0.25f, 0.73f, 0.999999f}) {
        const MediumSample sample =
            sampleMediumHomogeneous(sigmaT, sigmaS, distance, u);
        REQUIRE(!sample.scattered);
        requireNear(sample.t, distance, 0.0f, 0.0f,
                    "pure-absorption pass distance");
        requireSpectrumNear(sample.weight, expected, 2.0e-7f, 2.0e-6f,
                            "pure-absorption Beer-Lambert weight");
    }
}

void testOpticalCoefficientMapping() {
    CombustionFieldSample fields{};
    fields.density = 0.37f;
    fields.soot = 0.22f;
    fields.fuel = 0.61f;
    fields.reaction = 0.48f;
    fields.temperature = 1730.0f;
    const Spectrum smokeA =
        makeSpectrum(0.08f, 0.11f, 0.15f, 0.19f);
    const Spectrum smokeS =
        makeSpectrum(0.31f, 0.27f, 0.23f, 0.18f);
    const Spectrum sootA =
        makeSpectrum(0.44f, 0.52f, 0.61f, 0.70f);
    const Spectrum sootS =
        makeSpectrum(0.09f, 0.07f, 0.05f, 0.03f);
    const Spectrum flameA =
        makeSpectrum(0.02f, 0.04f, 0.06f, 0.08f);

    const Spectrum sigmaA =
        gridSigmaA(fields, smokeA, sootA, flameA);
    const Spectrum sigmaS =
        gridSigmaS(fields, smokeS, sootS);
    const Spectrum sigmaT = sigmaA + sigmaS;
    const Spectrum expectedA =
        smokeA * fields.density + sootA * fields.soot +
        flameA * fields.reaction;
    const Spectrum expectedS =
        smokeS * fields.density + sootS * fields.soot;
    requireSpectrumNear(sigmaA, expectedA, 1.0e-7f, 1.0e-6f,
                        "sigmaA field mapping");
    requireSpectrumNear(sigmaS, expectedS, 1.0e-7f, 1.0e-6f,
                        "sigmaS field mapping");

    for (int component = 0; component < kSpectrumComponentCount;
         ++component) {
        REQUIRE(sigmaA[component] >= 0.0f);
        REQUIRE(sigmaS[component] >= 0.0f);
        REQUIRE(sigmaT[component] >= 0.0f);
        requireNear(sigmaT[component],
                    sigmaA[component] + sigmaS[component], 1.0e-7f,
                    1.0e-7f, "sigmaT = sigmaA + sigmaS");
    }

    CombustionFieldSample empty{};
    REQUIRE(maxComponent(
                gridSigmaA(empty, smokeA, sootA, flameA)) == 0.0f);
    REQUIRE(maxComponent(gridSigmaS(empty, smokeS, sootS)) == 0.0f);
}

void testHenyeyGreensteinPhase() {
    constexpr int integrationSteps = 200000;
    constexpr float twoPi = 6.2831853071795864769f;
    const float dMu = 2.0f / static_cast<float>(integrationSteps);
    const std::array<float, 5> anisotropies = {
        -0.65f, 0.0f, 0.32f, 0.55f, 0.80f};

    for (float g : anisotropies) {
        double normalization = 0.0;
        double firstMoment = 0.0;
        double sampledMoment = 0.0;
        for (int i = 0; i < integrationSteps; ++i) {
            const float mu = -1.0f + (static_cast<float>(i) + 0.5f) * dMu;
            const float value = phaseHG(mu, g);
            REQUIRE(std::isfinite(value));
            REQUIRE(value >= 0.0f);
            normalization += static_cast<double>(value) * twoPi * dMu;
            firstMoment +=
                static_cast<double>(mu) * value * twoPi * dMu;

            const float u =
                (static_cast<float>(i) + 0.5f) /
                static_cast<float>(integrationSteps);
            const float sampledMu = sampleHGCosTheta(g, u);
            REQUIRE(std::isfinite(sampledMu));
            REQUIRE(sampledMu >= -1.0f && sampledMu <= 1.0f);
            sampledMoment += sampledMu;
        }

        requireNear(static_cast<float>(normalization), 1.0f, 2.5e-5f,
                    2.5e-5f, "HG phase normalization");
        requireNear(static_cast<float>(firstMoment), g, 6.0e-5f, 6.0e-5f,
                    "HG phase first moment");
        requireNear(
            static_cast<float>(sampledMoment / integrationSteps), g,
            6.0e-5f, 6.0e-5f, "HG inverse-CDF sample first moment");
    }
}

void testBlackbodyTemperatureRamp() {
    REQUIRE(planckSPD(550.0f, 0.0f) == 0.0f);

    // Independent numeric anchors from Planck's law, also used by PBRT-v4's
    // Apache-2.0 blackbody regression tests (commit 7154d826). Keeping these
    // physical-radiance checks catches unit mistakes that a qualitative color
    // ramp cannot, especially an accidental per-nm/per-metre factor of 1e9.
    struct PlanckReference {
        float wavelengthNm;
        float temperatureK;
        float radiancePerMetre;
    };
    const PlanckReference references[] = {
        {483.0f, 6000.0f, 3.1849e13f},
        {600.0f, 6000.0f, 2.86772e13f},
        {500.0f, 3700.0f, 1.59845e12f},
        {600.0f, 4500.0f, 7.46497e12f},
    };
    for (const PlanckReference &reference : references) {
        requireNear(planckSPD(reference.wavelengthNm,
                              reference.temperatureK),
                    reference.radiancePerMetre, 0.0f, 1.0e-3f,
                    "Planck spectral radiance");
    }

    const float coolRed = planckSPD(650.0f, 1200.0f);
    const float coolGreen = planckSPD(550.0f, 1200.0f);
    const float coolBlue = planckSPD(450.0f, 1200.0f);
    const float hotRed = planckSPD(650.0f, 3000.0f);
    const float hotGreen = planckSPD(550.0f, 3000.0f);
    const float hotBlue = planckSPD(450.0f, 3000.0f);
    for (float value :
         {coolRed, coolGreen, coolBlue, hotRed, hotGreen, hotBlue}) {
        REQUIRE(std::isfinite(value));
        REQUIRE(value > 0.0f);
    }
    REQUIRE(coolRed > coolGreen);
    REQUIRE(coolGreen > coolBlue);
    REQUIRE(hotGreen > coolGreen);
    REQUIRE(hotBlue > coolBlue);
    REQUIRE(hotBlue / hotRed > coolBlue / coolRed);
    REQUIRE(hotGreen / hotRed > coolGreen / coolRed);

    float previous = planckSPD(550.0f, 900.0f);
    for (float temperature :
         {1100.0f, 1400.0f, 1800.0f, 2300.0f, 3000.0f}) {
        const float value = planckSPD(550.0f, temperature);
        REQUIRE(value > previous);
        previous = value;
    }
}

template <typename Callable>
void requireInvalidArgument(Callable &&callable, const char *label) {
    try {
        callable();
    } catch (const std::invalid_argument &) {
        return;
    } catch (const std::exception &exception) {
        fail("throws std::invalid_argument", __FILE__, __LINE__,
             std::string(label) + " threw " + exception.what());
    }
    fail("throws std::invalid_argument", __FILE__, __LINE__, label);
}

void testMalformedResolutionThrows() {
    VolumeGridBuildSettings unaligned =
        defaultVolumeGridBuildSettings(CombustionPreset::Candle);
    unaligned.cellResolution = glm::ivec3(16, 17, 16);
    requireInvalidArgument(
        [&] { (void)buildSparseCombustionGrid(unaligned); },
        "unaligned resolution");

    VolumeGridBuildSettings partiallyZero =
        defaultVolumeGridBuildSettings(CombustionPreset::WoodFire);
    partiallyZero.cellResolution = glm::ivec3(0, 16, 16);
    requireInvalidArgument(
        [&] { (void)buildSparseCombustionGrid(partiallyZero); },
        "partially zero resolution");

    VolumeGridBuildSettings negative =
        defaultVolumeGridBuildSettings(CombustionPreset::Wildfire);
    negative.cellResolution = glm::ivec3(-8, 16, 16);
    requireInvalidArgument(
        [&] { (void)buildSparseCombustionGrid(negative); },
        "negative resolution");
}

} // namespace

int main() {
    using Test = std::pair<const char *, std::function<void()>>;
    const std::vector<Test> tests = {
        {"preset build and validation", testPresetBuildAndValidation},
        {"exact sparse trilinear interpolation",
         testExactSparseTrilinearInterpolation},
        {"conservative halo majorants", testConservativeHaloMajorants},
        {"page table and emission CDFs", testPageTableAndEmissionCdfs},
        {"emission CDF support", testEmissionCdfHasCompleteSupport},
        {"homogeneous analytic transmittance",
         testHomogeneousAnalyticTransmittance},
        {"homogeneous pure-absorption free flight",
         testHomogeneousPureAbsorptionFreeFlight},
        {"optical coefficient mapping", testOpticalCoefficientMapping},
        {"Henyey-Greenstein phase", testHenyeyGreensteinPhase},
        {"blackbody temperature ramp", testBlackbodyTemperatureRamp},
        {"malformed resolution rejection", testMalformedResolutionThrows},
    };

    int failures = 0;
    for (const Test &test : tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception &exception) {
            ++failures;
            std::cerr << "[FAIL] " << test.first << ": "
                      << exception.what() << '\n';
        }
    }

    if (failures == 0) {
        std::cout << "All " << tests.size()
                  << " sparse-volume tests passed.\n";
        return 0;
    }
    std::cerr << failures << " of " << tests.size()
              << " sparse-volume tests failed.\n";
    return 1;
}
