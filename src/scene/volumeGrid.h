#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>

// A brick contains 8^3 logical cells and a one-sample positive-side halo.
// Neighboring bricks deliberately duplicate their shared boundary samples.
// This makes trilinear filtering within a brick self-contained on the GPU.
constexpr int kVolumeBrickSize = 8;
constexpr int kVolumeBrickSampleSize = kVolumeBrickSize + 1;
constexpr int kVolumeBrickCellCount =
    kVolumeBrickSize * kVolumeBrickSize * kVolumeBrickSize;
constexpr int kVolumeBrickSampleCount =
    kVolumeBrickSampleSize * kVolumeBrickSampleSize *
    kVolumeBrickSampleSize;
// Must match the zero point of the incandescent-soot gate used by the
// renderer. A trilinear cell whose temperature exceeds this value and whose
// soot field is nonzero has potentially emissive support.
constexpr float kVolumeSootEmissionOnsetTemperature = 850.0f;

enum class CombustionPreset : int {
    Candle = 0,
    WoodFire = 1,
    Wildfire = 2,
};

// Construction controls for the deterministic, static combustion solve.
// cellResolution may be left at (0, 0, 0) to use the selected preset's
// production default. Explicit resolutions must be divisible by
// kVolumeBrickSize in every dimension.
struct VolumeGridBuildSettings {
    CombustionPreset preset = CombustionPreset::Candle;
    glm::ivec3 cellResolution = glm::ivec3(0);
    glm::vec3 localBoundsMin = glm::vec3(-0.5f);
    glm::vec3 localBoundsMax = glm::vec3(0.5f);
    glm::vec3 wind = glm::vec3(0.03f, 0.16f, 0.0f);
    int seed = 1;
    float activeThreshold = 0.0025f;
    float densityScale = 1.0f;
    float sootScale = 1.0f;
    float fuelScale = 1.0f;
    float temperatureScale = 1.0f;
    float reactionScale = 1.0f;
    float turbulenceScale = 1.0f;
    float buoyancyScale = 1.0f;
    float smokeAdvection = 1.0f;
    // Static showcase controls.  Values above one contract WoodFire fuel
    // sources around the grid center without shrinking the transported
    // smoke.  canopySpread adds a height-dependent radial component to the
    // WoodFire velocity field, allowing a compact burner to form a broad
    // soot canopy while all fields remain coupled in one grid.
    float sourceCompactness = 1.0f;
    float canopySpread = 0.0f;
    float canopyDensity = 0.0f;
};

// Metadata is copied byte-for-byte to CUDA memory. Maxima include all 9^3
// stored samples (including the duplicated halo), so trilinear interpolation
// cannot exceed maxFieldExtinction inside the brick when extinction is formed
// from density + soot with non-negative coefficients.
struct VolumeBrickMeta {
    int brickX = 0;
    int brickY = 0;
    int brickZ = 0;
    std::uint32_t sampleOffset = 0;
    std::uint32_t cellCdfOffset = 0;
    // Filled only in the device-upload copy. These sample-space atlas origins
    // avoid integer division/modulo at every hardware texture lookup.
    int textureBaseX = 0;
    int textureBaseY = 0;
    int textureBaseZ = 0;
    float maxDensity = 0.0f;
    float maxSoot = 0.0f;
    float maxFuel = 0.0f;
    float maxReaction = 0.0f;
    float maxTemperature = 0.0f;
    float maxFieldExtinction = 0.0f;
    float emissionPower = 0.0f;
};

// Host owner. pageTable is a dense brick-resolution array in x-major order;
// inactive entries are -1 and active entries index bricks. Sample layout is:
//   combustionSamples = (density, soot, fuel, reaction)
//   thermalFlowSamples = (temperature Kelvin, velocity.x/y/z)
// brickEmissionCdf is a normalized global CDF. cellEmissionCdf stores one
// normalized 8^3 CDF per active brick, at VolumeBrickMeta::cellCdfOffset.
struct HostSparseVolumeGrid {
    glm::ivec3 cellResolution = glm::ivec3(0);
    glm::ivec3 brickResolution = glm::ivec3(0);
    glm::vec3 localBoundsMin = glm::vec3(0.0f);
    glm::vec3 localBoundsMax = glm::vec3(0.0f);
    int brickSize = kVolumeBrickSize;
    float cellVolume = 0.0f;
    float totalEmissionPower = 0.0f;
    std::vector<int> pageTable;
    std::vector<VolumeBrickMeta> bricks;
    std::vector<glm::vec4> combustionSamples;
    std::vector<glm::vec4> thermalFlowSamples;
    std::vector<float> brickEmissionCdf;
    std::vector<float> cellEmissionCdf;
};

static_assert(std::is_trivially_copyable<VolumeBrickMeta>::value,
              "VolumeBrickMeta must be safe for cudaMemcpy");

VolumeGridBuildSettings
defaultVolumeGridBuildSettings(CombustionPreset preset);

HostSparseVolumeGrid
buildSparseCombustionGrid(const VolumeGridBuildSettings &settings);

// Performs structural, finite-value, CDF, page-table and conservative-majorant
// checks. On failure, the first diagnostic is written to error when non-null.
bool validateSparseCombustionGrid(const HostSparseVolumeGrid &grid,
                                  std::string *error = nullptr);
