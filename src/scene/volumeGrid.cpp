#include "volumeGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

constexpr float kAmbientTemperature = 293.15f;

struct NoiseSample {
    float value;
    glm::vec3 gradient;
};

struct SimulationCell {
    float density = 0.0f;
    float soot = 0.0f;
    float fuel = 0.0f;
    float reaction = 0.0f;
    float heat = 0.0f; // Kelvin above ambient
    float marker = 0.0f;
    glm::vec3 velocity = glm::vec3(0.0f);
};

struct SourceSample {
    float injection = 0.0f;
    float ignition = 0.0f;
    float solidFuel = 0.0f;
};

struct WoodBurner {
    glm::vec3 center;
    glm::vec3 radius;
    float strength;
    float heightScale;
};

// Fuel-release pockets mapped from the upper surfaces and intersections of
// the three logs in showcase_wood_fire.json. Both the combustion source and
// transported flame reconstruction consume this table, keeping authored root
// position, height, and optical emission registered while weak pairwise
// overlap joins neighboring gaps without forming a common base plane.
const std::array<WoodBurner, 8> kWoodBurners = {{
    {glm::vec3(0.245f, 0.060f, 0.350f),
     glm::vec3(0.036f, 0.030f, 0.068f), 0.68f, 0.72f},
    {glm::vec3(0.345f, 0.082f, 0.680f),
     glm::vec3(0.040f, 0.032f, 0.074f), 0.78f, 1.15f},
    {glm::vec3(0.425f, 0.112f, 0.450f),
     glm::vec3(0.044f, 0.034f, 0.080f), 0.96f, 0.82f},
    // Camera-side gap on the upper surface of the crossing log.
    {glm::vec3(0.505f, 0.110f, 0.950f),
     glm::vec3(0.044f, 0.038f, 0.078f), 0.70f, 1.32f},
    {glm::vec3(0.560f, 0.142f, 0.350f),
     glm::vec3(0.044f, 0.035f, 0.080f), 0.98f, 0.95f},
    {glm::vec3(0.650f, 0.092f, 0.620f),
     glm::vec3(0.040f, 0.032f, 0.074f), 0.80f, 1.12f},
    {glm::vec3(0.735f, 0.068f, 0.420f),
     glm::vec3(0.036f, 0.030f, 0.068f), 0.66f, 0.65f},
    // A second camera-side tongue wraps the right-hand log.
    {glm::vec3(0.680f, 0.090f, 0.950f),
     glm::vec3(0.040f, 0.033f, 0.075f), 0.58f, 0.88f},
}};

struct PresetParameters {
    int steps;
    float dt;
    float baseRise;
    float fuelRetention;
    float reactionRetention;
    float heatRetention;
    float densityRetention;
    float sootRetention;
    float injectionRate;
    float burnRate;
    float reactionGain;
    float heatYield;
    float sourceHeat;
    float gasYield;
    float sootYield;
    float maximumHeat;
};

inline float saturate(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

inline float smoothstep(float edge0, float edge1, float value) {
    if (edge0 == edge1) {
        return value < edge0 ? 0.0f : 1.0f;
    }
    const float t = saturate((value - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

inline float square(float value) { return value * value; }

inline bool finiteVec3(const glm::vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

inline bool finiteVec4(const glm::vec4 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

std::uint32_t mixBits(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float latticeValue(int x, int y, int z, int seed) {
    const std::uint32_t h =
        mixBits(static_cast<std::uint32_t>(x) * 0x8da6b343u ^
                static_cast<std::uint32_t>(y) * 0xd8163841u ^
                static_cast<std::uint32_t>(z) * 0xcb1ab31fu ^
                static_cast<std::uint32_t>(seed) * 0x9e3779b9u);
    return static_cast<float>(h & 0x00ffffffu) / 8388607.5f - 1.0f;
}

// Smooth value noise with an analytic derivative. The derivative makes it
// inexpensive to construct curl of a vector potential without finite
// differencing six additional noise evaluations.
NoiseSample valueNoiseDerivative(const glm::vec3 &position, int seed) {
    const glm::vec3 baseFloat = glm::floor(position);
    const glm::ivec3 base(baseFloat);
    const glm::vec3 f = position - baseFloat;
    const glm::vec3 fade =
        f * f * (glm::vec3(3.0f) - glm::vec3(2.0f) * f);
    const glm::vec3 fadeDerivative =
        glm::vec3(6.0f) * f * (glm::vec3(1.0f) - f);

    NoiseSample result{0.0f, glm::vec3(0.0f)};
    for (int z = 0; z <= 1; ++z) {
        const float wz = z ? fade.z : 1.0f - fade.z;
        const float dwz = z ? fadeDerivative.z : -fadeDerivative.z;
        for (int y = 0; y <= 1; ++y) {
            const float wy = y ? fade.y : 1.0f - fade.y;
            const float dwy = y ? fadeDerivative.y : -fadeDerivative.y;
            for (int x = 0; x <= 1; ++x) {
                const float wx = x ? fade.x : 1.0f - fade.x;
                const float dwx =
                    x ? fadeDerivative.x : -fadeDerivative.x;
                const float value =
                    latticeValue(base.x + x, base.y + y, base.z + z,
                                 seed);
                result.value += value * wx * wy * wz;
                result.gradient.x += value * dwx * wy * wz;
                result.gradient.y += value * wx * dwy * wz;
                result.gradient.z += value * wx * wy * dwz;
            }
        }
    }
    return result;
}

float fractalNoise(const glm::vec3 &position, int seed) {
    float value = 0.0f;
    float normalization = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    for (int octave = 0; octave < 4; ++octave) {
        value += amplitude *
                 valueNoiseDerivative(position * frequency,
                                      seed + octave * 101).value;
        normalization += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.03f;
    }
    return value / normalization;
}

// Curl of a two-octave vector-potential field. This is divergence-free up to
// interpolation error and produces coherent rolls rather than independent
// world-space density noise.
glm::vec3 curlNoise(const glm::vec3 &position, int seed) {
    glm::vec3 curl(0.0f);
    float amplitude = 0.72f;
    float frequency = 2.1f;
    for (int octave = 0; octave < 2; ++octave) {
        const glm::vec3 p = position * frequency;
        const NoiseSample ax =
            valueNoiseDerivative(p + glm::vec3(13.2f, 1.7f, 9.1f),
                                 seed + octave * 131 + 11);
        const NoiseSample ay =
            valueNoiseDerivative(p + glm::vec3(3.8f, 17.3f, 5.4f),
                                 seed + octave * 131 + 37);
        const NoiseSample az =
            valueNoiseDerivative(p + glm::vec3(7.5f, 11.6f, 19.8f),
                                 seed + octave * 131 + 73);
        const glm::vec3 octaveCurl(
            az.gradient.y - ay.gradient.z,
            ax.gradient.z - az.gradient.x,
            ay.gradient.x - ax.gradient.y);
        curl += octaveCurl * (amplitude * frequency);
        amplitude *= 0.43f;
        frequency *= 2.17f;
    }
    const float magnitude = glm::length(curl);
    return magnitude > 0.0f ? curl / (1.0f + 0.55f * magnitude)
                            : glm::vec3(0.0f);
}

std::size_t sampleIndex(const glm::ivec3 &sampleResolution, int x, int y,
                        int z) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(sampleResolution.x) *
               (static_cast<std::size_t>(y) +
                static_cast<std::size_t>(sampleResolution.y) *
                    static_cast<std::size_t>(z));
}

std::size_t pageIndex(const glm::ivec3 &brickResolution, int x, int y,
                      int z) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(brickResolution.x) *
               (static_cast<std::size_t>(y) +
                static_cast<std::size_t>(brickResolution.y) *
                    static_cast<std::size_t>(z));
}

int brickSampleIndex(int x, int y, int z) {
    return x + kVolumeBrickSampleSize *
                   (y + kVolumeBrickSampleSize * z);
}

int brickCellIndex(int x, int y, int z) {
    return x + kVolumeBrickSize * (y + kVolumeBrickSize * z);
}

SimulationCell lerpCell(const SimulationCell &a,
                        const SimulationCell &b, float t) {
    SimulationCell result;
    result.density = a.density + (b.density - a.density) * t;
    result.soot = a.soot + (b.soot - a.soot) * t;
    result.fuel = a.fuel + (b.fuel - a.fuel) * t;
    result.reaction = a.reaction + (b.reaction - a.reaction) * t;
    result.heat = a.heat + (b.heat - a.heat) * t;
    result.marker = a.marker + (b.marker - a.marker) * t;
    result.velocity = a.velocity + (b.velocity - a.velocity) * t;
    return result;
}

SimulationCell sampleSimulation(const std::vector<SimulationCell> &field,
                                const glm::ivec3 &cellResolution,
                                const glm::vec3 &normalizedPosition) {
    if (normalizedPosition.x < 0.0f || normalizedPosition.y < 0.0f ||
        normalizedPosition.z < 0.0f || normalizedPosition.x > 1.0f ||
        normalizedPosition.y > 1.0f || normalizedPosition.z > 1.0f) {
        return SimulationCell{};
    }

    const glm::ivec3 sampleResolution = cellResolution + glm::ivec3(1);
    const glm::vec3 gridPosition =
        normalizedPosition * glm::vec3(cellResolution);
    glm::ivec3 lower(glm::floor(gridPosition));
    lower = glm::clamp(lower, glm::ivec3(0),
                       cellResolution - glm::ivec3(1));
    const glm::vec3 fraction = gridPosition - glm::vec3(lower);
    const glm::ivec3 upper = lower + glm::ivec3(1);

    const SimulationCell c000 =
        field[sampleIndex(sampleResolution, lower.x, lower.y, lower.z)];
    const SimulationCell c100 =
        field[sampleIndex(sampleResolution, upper.x, lower.y, lower.z)];
    const SimulationCell c010 =
        field[sampleIndex(sampleResolution, lower.x, upper.y, lower.z)];
    const SimulationCell c110 =
        field[sampleIndex(sampleResolution, upper.x, upper.y, lower.z)];
    const SimulationCell c001 =
        field[sampleIndex(sampleResolution, lower.x, lower.y, upper.z)];
    const SimulationCell c101 =
        field[sampleIndex(sampleResolution, upper.x, lower.y, upper.z)];
    const SimulationCell c011 =
        field[sampleIndex(sampleResolution, lower.x, upper.y, upper.z)];
    const SimulationCell c111 =
        field[sampleIndex(sampleResolution, upper.x, upper.y, upper.z)];

    const SimulationCell c00 = lerpCell(c000, c100, fraction.x);
    const SimulationCell c10 = lerpCell(c010, c110, fraction.x);
    const SimulationCell c01 = lerpCell(c001, c101, fraction.x);
    const SimulationCell c11 = lerpCell(c011, c111, fraction.x);
    const SimulationCell c0 = lerpCell(c00, c10, fraction.y);
    const SimulationCell c1 = lerpCell(c01, c11, fraction.y);
    return lerpCell(c0, c1, fraction.z);
}

float sampleScalar(const std::vector<float> &field,
                   const glm::ivec3 &cellResolution,
                   const glm::vec3 &normalizedPosition) {
    if (normalizedPosition.x < 0.0f || normalizedPosition.y < 0.0f ||
        normalizedPosition.z < 0.0f || normalizedPosition.x > 1.0f ||
        normalizedPosition.y > 1.0f || normalizedPosition.z > 1.0f) {
        return 0.0f;
    }

    const glm::ivec3 sampleResolution =
        cellResolution + glm::ivec3(1);
    const glm::vec3 gridPosition =
        normalizedPosition * glm::vec3(cellResolution);
    glm::ivec3 lower(glm::floor(gridPosition));
    lower = glm::clamp(lower, glm::ivec3(0),
                       cellResolution - glm::ivec3(1));
    const glm::vec3 fraction =
        gridPosition - glm::vec3(lower);
    const glm::ivec3 upper = lower + glm::ivec3(1);
    const auto at = [&](int x, int y, int z) {
        return field[sampleIndex(sampleResolution, x, y, z)];
    };
    const float x00 = glm::mix(at(lower.x, lower.y, lower.z),
                               at(upper.x, lower.y, lower.z),
                               fraction.x);
    const float x10 = glm::mix(at(lower.x, upper.y, lower.z),
                               at(upper.x, upper.y, lower.z),
                               fraction.x);
    const float x01 = glm::mix(at(lower.x, lower.y, upper.z),
                               at(upper.x, lower.y, upper.z),
                               fraction.x);
    const float x11 = glm::mix(at(lower.x, upper.y, upper.z),
                               at(upper.x, upper.y, upper.z),
                               fraction.x);
    return glm::mix(glm::mix(x00, x10, fraction.y),
                    glm::mix(x01, x11, fraction.y), fraction.z);
}

float gaussianEllipsoid(const glm::vec3 &position,
                        const glm::vec3 &center,
                        const glm::vec3 &radius) {
    const glm::vec3 d = (position - center) / radius;
    return std::exp(-glm::dot(d, d));
}

// A static, velocity-aligned upper smoke reconstruction for a compact wood
// burner.  It is baked into the same sparse fields as the transported gas,
// rather than evaluated by the renderer.  Strongly overlapping updraft
// kernels form one continuous canopy; coherent 3-D domain warping and
// erosion provide folds, dense lobes, and translucent edges without a
// union-of-spheres silhouette.  The default amplitude is zero, preserving
// the production wood-fire preset unless a scene explicitly requests a
// larger soot release.
float woodSootCanopyAt(const glm::vec3 &q,
                       const VolumeGridBuildSettings &settings) {
    if (settings.canopyDensity <= 0.0f ||
        settings.preset != CombustionPreset::WoodFire) {
        return 0.0f;
    }

    const float h = q.y;
    const float warpX =
        fractalNoise(glm::vec3(q.x * 2.7f, h * 2.1f,
                               q.z * 2.5f),
                     settings.seed + 3109);
    const float warpZ =
        fractalNoise(glm::vec3(q.x * 2.3f + 7.1f, h * 2.4f,
                               q.z * 2.9f + 3.7f),
                     settings.seed + 3251);
    const float rise = smoothstep(0.22f, 0.86f, h);
    glm::vec3 p = q;
    p.x -= settings.wind.x * (0.16f + 0.22f * rise);
    p.z -= settings.wind.z * (0.16f + 0.22f * rise);
    p.x += (0.030f + 0.085f * rise) * warpX;
    p.z += (0.026f + 0.075f * rise) * warpZ;

    const float stemRadius = 0.035f + 0.105f *
        smoothstep(0.18f, 0.72f, h);
    const float stemX = 0.50f + 0.055f *
        std::sin(8.1f * h + 1.7f * warpX);
    const float stemZ = 0.50f + 0.045f *
        std::cos(7.3f * h - 1.5f * warpZ);
    const float stemR2 =
        square((p.x - stemX) / stemRadius) +
        square((p.z - stemZ) / (0.82f * stemRadius));
    float stem = std::exp(-stemR2) *
        smoothstep(0.14f, 0.30f, h) *
        (1.0f - smoothstep(0.76f, 0.94f, h));

    float sum = 0.0f;
    sum += 1.15f * gaussianEllipsoid(
        p, glm::vec3(0.27f, 0.68f, 0.50f),
        glm::vec3(0.27f, 0.18f, 0.26f));
    sum += 1.36f * gaussianEllipsoid(
        p, glm::vec3(0.45f, 0.73f, 0.47f),
        glm::vec3(0.28f, 0.20f, 0.28f));
    sum += 1.22f * gaussianEllipsoid(
        p, glm::vec3(0.67f, 0.60f, 0.54f),
        glm::vec3(0.25f, 0.17f, 0.25f));
    sum += 0.72f * gaussianEllipsoid(
        p, glm::vec3(0.34f, 0.84f, 0.45f),
        glm::vec3(0.22f, 0.13f, 0.22f));
    const float mass = 1.0f - std::exp(-sum);

    const float broad =
        fractalNoise(glm::vec3(p.x * 4.1f, p.y * 3.6f,
                               p.z * 4.0f),
                     settings.seed + 3413);
    const float detail =
        fractalNoise(glm::vec3(p.x * 10.7f, p.y * 7.2f,
                               p.z * 9.8f),
                     settings.seed + 3559);
    const float silhouette =
        fractalNoise(glm::vec3(p.x * 5.2f, p.y * 4.7f,
                               17.3f),
                     settings.seed + 3701);
    const float eroded =
        smoothstep(0.18f, 0.74f,
                   mass + 0.23f * broad + 0.12f * detail +
                       0.13f * silhouette);
    const float internal =
        0.055f + 0.945f * smoothstep(-0.48f, 0.58f,
                                     0.58f * broad + 0.27f * detail +
                                         0.15f * silhouette);
    float canopy = eroded * internal;
    const float baseVariation =
        0.045f * broad + 0.030f * silhouette;
    canopy *= smoothstep(0.42f + baseVariation,
                         0.52f + baseVariation, h) *
              (1.0f - smoothstep(0.88f + 0.035f * silhouette,
                                 0.985f, h));
    canopy *= 0.12f + 0.88f *
        smoothstep(-0.56f, 0.48f, silhouette + 0.35f * mass);
    stem *= 0.44f + 0.56f * smoothstep(-0.72f, 0.38f, broad);
    return saturate(std::max(canopy, 0.78f * stem));
}

float segmentDistanceSquared(const glm::vec3 &position,
                             const glm::vec3 &start,
                             const glm::vec3 &end) {
    const glm::vec3 delta = end - start;
    const float denominator = glm::dot(delta, delta);
    const float t = denominator > 0.0f
                        ? saturate(glm::dot(position - start, delta) /
                                   denominator)
                        : 0.0f;
    const glm::vec3 nearest = start + t * delta;
    return glm::dot(position - nearest, position - nearest);
}

float wildfireFrontZAt(float x, int seed) {
    const float frontLarge =
        fractalNoise(glm::vec3(x * 3.2f, 2.1f, 7.4f), seed + 151);
    const float frontFine =
        fractalNoise(glm::vec3(x * 8.7f, 5.3f, 1.8f), seed + 313);
    return 0.27f + 0.085f * frontLarge + 0.025f * frontFine;
}

SourceSample sourceAt(const glm::vec3 &q,
                      const VolumeGridBuildSettings &settings) {
    SourceSample source;
    const CombustionPreset preset = settings.preset;
    const int seed = settings.seed;

    if (preset == CombustionPreset::Candle) {
        const float vapor =
            gaussianEllipsoid(q, glm::vec3(0.5f, 0.055f, 0.5f),
                              glm::vec3(0.090f, 0.060f, 0.078f));
        const float wick =
            gaussianEllipsoid(q, glm::vec3(0.5f, 0.022f, 0.5f),
                              glm::vec3(0.038f, 0.034f, 0.034f));
        source.injection = saturate(1.18f * vapor);
        source.ignition = saturate(std::max(vapor, 1.35f * wick));
        source.solidFuel = saturate(wick);
        return source;
    }

    if (preset == CombustionPreset::WoodFire) {
        glm::vec3 sourceQ = q;
        sourceQ.x =
            (q.x - 0.5f) * settings.sourceCompactness + 0.5f;
        sourceQ.z =
            (q.z - 0.5f) * settings.sourceCompactness + 0.5f;
        float localizedZones = 0.0f;
        for (const WoodBurner &burner : kWoodBurners) {
            localizedZones =
                std::max(
                    localizedZones,
                    burner.strength *
                        gaussianEllipsoid(sourceQ, burner.center,
                                          burner.radius));
        }
        localizedZones = saturate(localizedZones);

        // Thin fuel-release zones follow two crossed log surfaces. They fill
        // gaps between the localized burners without creating a base plane.
        const float lineA = std::exp(
            -segmentDistanceSquared(
                 sourceQ, glm::vec3(0.18f, 0.075f, 0.31f),
                 glm::vec3(0.80f, 0.115f, 0.67f)) /
            square(0.058f));
        const float lineB = std::exp(
            -segmentDistanceSquared(
                 sourceQ, glm::vec3(0.19f, 0.095f, 0.69f),
                 glm::vec3(0.78f, 0.075f, 0.32f)) /
            square(0.054f));
        const float logSurface = std::max(lineA, lineB);
        const float breakup =
            0.62f + 0.38f * saturate(0.5f +
                                     0.5f * fractalNoise(sourceQ * 8.0f, seed + 9));
        const float fuelPocket =
            gaussianEllipsoid(sourceQ, glm::vec3(0.50f, 0.085f, 0.50f),
                              glm::vec3(0.35f, 0.060f, 0.24f)) *
            smoothstep(
                -0.72f, 0.36f,
                fractalNoise(sourceQ * glm::vec3(6.2f, 4.1f, 7.5f),
                             seed + 117));
        // The long log-surface curves hold fuel, but only sparse hot pockets
        // release and ignite it.  Letting the full curves ignite was the
        // source of the visually planar flame curtain behind the logs.
        const float pocketedSurface =
            logSurface *
            smoothstep(0.48f, 0.80f, breakup) *
            smoothstep(
                0.42f, 0.78f,
                0.5f +
                    0.5f * fractalNoise(
                               sourceQ * glm::vec3(12.0f, 5.0f, 10.5f),
                               seed + 1301));
        source.injection = saturate(
            (1.02f * localizedZones + 0.055f * pocketedSurface +
             0.035f * fuelPocket) *
            breakup);
        source.ignition =
            saturate(localizedZones + 0.045f * pocketedSurface +
                     0.025f * fuelPocket);
        source.solidFuel =
            saturate(0.58f * logSurface + 0.12f * fuelPocket);
        return source;
    }

    // A continuous but irregular ground-attached wildfire front. q.x runs
    // along the front; positive q.z is downwind in the default preset.
    const float frontZ = wildfireFrontZAt(q.x, seed);
    const float distanceToFront = std::abs(q.z - frontZ);
    // Keep the combustion sheet slightly above the lower grid boundary.
    // Centering it at y=0 discarded half of the source and let an opaque
    // ground surface clip the hottest cells in the showcase.
    const float fuelHeight = q.y - 0.035f;
    const float ground = std::exp(-square(fuelHeight / 0.065f));
    const float fuelVariation =
        smoothstep(-0.55f, 0.65f,
                   fractalNoise(glm::vec3(q.x * 5.4f, q.z * 5.4f, 3.7f),
                                seed + 47));
    const float fineFuel =
        smoothstep(-0.70f, 0.72f,
                   fractalNoise(glm::vec3(q.x * 13.0f, q.z * 13.0f, 9.2f),
                                seed + 83));
    const float vegetationFuel =
        saturate(0.30f + 0.47f * fuelVariation + 0.23f * fineFuel);
    const float mainFrontBand =
        std::exp(-square(distanceToFront /
                         (0.035f + 0.022f * vegetationFuel)));
    // All wildfire support comes from the coherently warped main front and
    // spatially varying vegetation fuel. No authored branch geometry or
    // localized flame placement participates in the source field.
    const float frontBand = mainFrontBand;
    const float nearbyFuel =
        std::exp(-square(fuelHeight / 0.040f)) * vegetationFuel;

    // The nonzero floor keeps the front connected; coherent fuel variation
    // controls intensity and tall flare-ups rather than spacing ribbons.
    source.injection =
        saturate(frontBand * ground * (0.62f + 0.62f * vegetationFuel));
    source.ignition =
        saturate(frontBand * ground * (0.78f + 0.38f * vegetationFuel));
    source.solidFuel = saturate(nearbyFuel);
    return source;
}

PresetParameters presetParameters(CombustionPreset preset) {
    if (preset == CombustionPreset::Candle) {
        return {38,   0.020f, 0.48f, 0.84f, 0.68f, 0.944f, 0.972f,
                0.979f, 0.45f, 0.42f, 4.2f, 1820.0f, 48.0f, 0.26f,
                0.16f, 2200.0f};
    }
    if (preset == CombustionPreset::WoodFire) {
        return {40,   0.019f, 0.55f, 0.87f, 0.50f, 0.950f, 0.980f,
                0.989f, 0.38f, 0.40f, 4.6f, 1950.0f, 44.0f, 0.35f,
                0.39f, 2450.0f};
    }
    return {48,   0.017f, 0.45f, 0.86f, 0.48f, 0.948f, 0.990f,
            0.996f, 0.37f, 0.38f, 4.0f, 1960.0f, 42.0f, 0.48f,
            0.72f, 2400.0f};
}

glm::vec3 baseVelocityAt(const glm::vec3 &q,
                         const VolumeGridBuildSettings &settings,
                         CombustionPreset preset,
                         const glm::vec3 &curl) {
    const PresetParameters parameters = presetParameters(preset);
    glm::vec3 velocity = settings.wind;
    const glm::vec2 fromCenter(q.x - 0.5f, q.z - 0.5f);

    if (preset == CombustionPreset::Candle) {
        const float core =
            std::exp(-glm::dot(fromCenter, fromCenter) / square(0.20f));
        velocity.x += -0.15f * fromCenter.x * (1.0f - q.y);
        velocity.z += -0.15f * fromCenter.y * (1.0f - q.y);
        velocity.y += parameters.baseRise * (0.50f + 0.64f * core);
        velocity += curl * (0.055f + 0.15f * q.y) *
                    settings.turbulenceScale;
    } else if (preset == CombustionPreset::WoodFire) {
        const float core =
            std::exp(-glm::dot(fromCenter, fromCenter) / square(0.36f));
        const float localizedUpdraft =
            smoothstep(
                -0.52f, 0.64f,
                fractalNoise(
                    glm::vec3(q.x * 5.8f, q.y * 2.2f,
                              q.z * 5.1f),
                    settings.seed + 811)) *
            std::exp(-q.y / 0.62f);
        velocity.x += -0.10f * fromCenter.x * (1.0f - q.y);
        velocity.z += -0.10f * fromCenter.y * (1.0f - q.y);
        velocity.y += parameters.baseRise *
                      (0.32f + 0.28f * core +
                       0.52f * localizedUpdraft);
        const float canopy = smoothstep(0.18f, 0.86f, q.y);
        velocity.x += settings.canopySpread * fromCenter.x * canopy;
        velocity.z += settings.canopySpread * fromCenter.y * canopy;
        velocity += curl * (0.18f + 0.30f * q.y) *
                    settings.turbulenceScale;
    } else {
        const float nearFront =
            std::exp(-square((q.z - 0.29f) / 0.24f));
        const float flareUp =
            smoothstep(
                0.58f, 0.88f,
                fractalNoise(
                    glm::vec3(q.x * 7.1f, 3.7f, q.z * 4.2f),
                    settings.seed + 733)) *
            std::exp(-square(q.y / 0.38f));
        velocity.y += parameters.baseRise *
                      (0.38f + 0.60f * nearFront) *
                      (1.0f - 0.35f * q.y);
        velocity.y += 0.34f * flareUp * nearFront;
        velocity += curl * (0.16f + 0.22f * q.y) *
                    settings.turbulenceScale;
        // Expanding hot gases turn the coherent wind into a broad, leaning
        // smoke canopy rather than a collection of parallel columns.
        velocity.x += 0.07f * (q.x - 0.5f) * q.y;
        velocity.z += 0.10f * q.y;
    }
    return velocity;
}

float emissionProxy(
    const std::array<glm::vec4, kVolumeBrickSampleCount> &combustion,
    const std::array<glm::vec4, kVolumeBrickSampleCount> &thermalFlow,
    int cellX, int cellY, int cellZ) {
    glm::vec4 averageCombustion(0.0f);
    glm::vec4 averageThermal(0.0f);
    float maxReaction = 0.0f;
    float maxSoot = 0.0f;
    float maxTemperature = 0.0f;
    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                const int sample =
                    brickSampleIndex(cellX + dx, cellY + dy, cellZ + dz);
                const glm::vec4 c = combustion[sample];
                const glm::vec4 t = thermalFlow[sample];
                averageCombustion += c;
                averageThermal += t;
                maxReaction = std::max(maxReaction, c.w);
                maxSoot = std::max(maxSoot, c.y);
                maxTemperature = std::max(maxTemperature, t.x);
            }
        }
    }
    averageCombustion *= 0.125f;
    averageThermal *= 0.125f;

    // A stable Stefan-Boltzmann-like importance proxy. Actual renderer
    // emission remains spectral blackbody; this only constructs a sampling
    // distribution and therefore need not encode flame color.
    const float normalizedTemperature =
        std::max(0.0f, (averageThermal.x - 650.0f) / 1450.0f);
    const float thermalPower =
        square(square(std::min(normalizedTemperature, 2.0f)));
    const float emissiveMatter =
        std::max(0.0f, averageCombustion.w) +
        0.42f * std::max(0.0f, averageCombustion.y) *
            saturate(normalizedTemperature);
    float proxy = thermalPower * emissiveMatter;

    // Corner averages are a good importance estimate but not a conservative
    // support test. Runtime reaction emission has no temperature gate, and
    // separately interpolated soot and temperature fields can overlap in the
    // cell interior even when their positive maxima occur at different
    // corners. Give every potentially emissive trilinear cell a small,
    // representable sampling floor. This modifies only the proposal
    // distribution; gridEmissionSource remains the source of actual emitted
    // radiance and therefore the estimator stays unbiased.
    const bool hasReactionSupport = maxReaction > 0.0f;
    const bool hasSootSupport =
        maxSoot > 0.0f &&
        maxTemperature > kVolumeSootEmissionOnsetTemperature;
    if (hasReactionSupport || hasSootSupport) {
        constexpr float kEmissionSupportProxyFloor = 1.0e-3f;
        proxy = std::max(proxy, kEmissionSupportProxyFloor);
    }
    return proxy;
}

// Expand every active brick's field bounds by one brick in all directions.
// The update reads from a snapshot so this is exactly one Chebyshev-radius
// dilation (the brick itself plus its 26 neighbors), not a traversal-order
// dependent flood. Only majorant/debug maxima are changed: sample offsets,
// CDF offsets, emitted power and all field payloads remain bit-identical.
void dilateBrickMaxima26(HostSparseVolumeGrid &grid) {
    if (grid.bricks.empty()) {
        return;
    }

    const std::vector<VolumeBrickMeta> undilated = grid.bricks;
    for (std::size_t brickIndex = 0; brickIndex < grid.bricks.size();
         ++brickIndex) {
        VolumeBrickMeta &destination = grid.bricks[brickIndex];
        const VolumeBrickMeta &center = undilated[brickIndex];
        for (int dz = -1; dz <= 1; ++dz) {
            const int z = center.brickZ + dz;
            if (z < 0 || z >= grid.brickResolution.z) {
                continue;
            }
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = center.brickY + dy;
                if (y < 0 || y >= grid.brickResolution.y) {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = center.brickX + dx;
                    if (x < 0 || x >= grid.brickResolution.x) {
                        continue;
                    }
                    const int neighborIndex =
                        grid.pageTable[pageIndex(grid.brickResolution, x,
                                                 y, z)];
                    if (neighborIndex < 0) {
                        continue;
                    }
                    const VolumeBrickMeta &neighbor =
                        undilated[static_cast<std::size_t>(
                            neighborIndex)];
                    destination.maxDensity =
                        std::max(destination.maxDensity,
                                 neighbor.maxDensity);
                    destination.maxSoot =
                        std::max(destination.maxSoot, neighbor.maxSoot);
                    destination.maxFuel =
                        std::max(destination.maxFuel, neighbor.maxFuel);
                    destination.maxReaction =
                        std::max(destination.maxReaction,
                                 neighbor.maxReaction);
                    destination.maxTemperature =
                        std::max(destination.maxTemperature,
                                 neighbor.maxTemperature);
                    destination.maxFieldExtinction =
                        std::max(destination.maxFieldExtinction,
                                 neighbor.maxFieldExtinction);
                }
            }
        }
    }
}

bool failValidation(std::string *error, const std::string &message) {
    if (error != nullptr) {
        *error = message;
    }
    return false;
}

bool validateNormalizedCdf(const float *begin, std::size_t count,
                           const std::string &name,
                           std::string *error) {
    if (count == 0) {
        return true;
    }
    float previous = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float value = begin[i];
        if (!std::isfinite(value) || value < previous - 1.0e-6f ||
            value < -1.0e-6f || value > 1.0f + 1.0e-5f) {
            std::ostringstream stream;
            stream << name << " is invalid at index " << i;
            return failValidation(error, stream.str());
        }
        previous = value;
    }
    if (std::abs(previous - 1.0f) > 2.0e-5f) {
        return failValidation(error, name + " does not terminate at 1");
    }
    return true;
}

} // namespace

VolumeGridBuildSettings
defaultVolumeGridBuildSettings(CombustionPreset preset) {
    VolumeGridBuildSettings settings;
    settings.preset = preset;

    if (preset == CombustionPreset::Candle) {
        settings.cellResolution = glm::ivec3(48, 96, 48);
        settings.wind = glm::vec3(0.025f, 0.14f, -0.005f);
        settings.activeThreshold = 0.0020f;
        settings.densityScale = 0.72f;
        settings.sootScale = 0.30f;
        settings.temperatureScale = 1.04f;
        settings.turbulenceScale = 0.72f;
        settings.buoyancyScale = 1.08f;
        settings.smokeAdvection = 1.05f;
    } else if (preset == CombustionPreset::WoodFire) {
        settings.cellResolution = glm::ivec3(80, 96, 64);
        settings.wind = glm::vec3(0.055f, 0.10f, 0.018f);
        settings.activeThreshold = 0.0025f;
        settings.densityScale = 1.00f;
        settings.sootScale = 0.82f;
        settings.temperatureScale = 1.02f;
        settings.reactionScale = 1.08f;
        settings.turbulenceScale = 1.15f;
        settings.buoyancyScale = 1.05f;
        settings.smokeAdvection = 1.13f;
    } else {
        settings.cellResolution = glm::ivec3(128, 64, 80);
        settings.wind = glm::vec3(0.10f, 0.055f, 0.82f);
        settings.activeThreshold = 0.0022f;
        settings.densityScale = 1.16f;
        settings.sootScale = 1.28f;
        settings.temperatureScale = 0.98f;
        settings.reactionScale = 1.12f;
        settings.turbulenceScale = 1.24f;
        settings.buoyancyScale = 0.95f;
        settings.smokeAdvection = 1.34f;
    }
    return settings;
}

HostSparseVolumeGrid
buildSparseCombustionGrid(const VolumeGridBuildSettings &inputSettings) {
    VolumeGridBuildSettings settings = inputSettings;
    const int presetValue = static_cast<int>(settings.preset);
    if (presetValue < static_cast<int>(CombustionPreset::Candle) ||
        presetValue > static_cast<int>(CombustionPreset::Wildfire)) {
        throw std::invalid_argument("unknown combustion volume preset");
    }
    if (settings.cellResolution == glm::ivec3(0)) {
        const VolumeGridBuildSettings defaults =
            defaultVolumeGridBuildSettings(settings.preset);
        settings.cellResolution = defaults.cellResolution;
    }

    if (settings.cellResolution.x <= 0 ||
        settings.cellResolution.y <= 0 ||
        settings.cellResolution.z <= 0) {
        throw std::invalid_argument(
            "volume cell resolution must be positive in all dimensions");
    }
    if (settings.cellResolution.x % kVolumeBrickSize != 0 ||
        settings.cellResolution.y % kVolumeBrickSize != 0 ||
        settings.cellResolution.z % kVolumeBrickSize != 0) {
        throw std::invalid_argument(
            "volume cell resolution must be divisible by brick size 8");
    }
    if (!finiteVec3(settings.localBoundsMin) ||
        !finiteVec3(settings.localBoundsMax) ||
        glm::any(glm::lessThanEqual(settings.localBoundsMax,
                                    settings.localBoundsMin))) {
        throw std::invalid_argument("volume local bounds are invalid");
    }
    if (!finiteVec3(settings.wind) ||
        !std::isfinite(settings.activeThreshold) ||
        settings.activeThreshold < 0.0f ||
        !std::isfinite(settings.densityScale) ||
        settings.densityScale < 0.0f ||
        !std::isfinite(settings.sootScale) || settings.sootScale < 0.0f ||
        !std::isfinite(settings.fuelScale) || settings.fuelScale < 0.0f ||
        !std::isfinite(settings.temperatureScale) ||
        settings.temperatureScale < 0.0f ||
        !std::isfinite(settings.reactionScale) ||
        settings.reactionScale < 0.0f ||
        !std::isfinite(settings.turbulenceScale) ||
        settings.turbulenceScale < 0.0f ||
        !std::isfinite(settings.buoyancyScale) ||
        settings.buoyancyScale < 0.0f ||
        !std::isfinite(settings.smokeAdvection) ||
        settings.smokeAdvection <= 0.0f ||
        !std::isfinite(settings.sourceCompactness) ||
        settings.sourceCompactness <= 0.0f ||
        !std::isfinite(settings.canopySpread) ||
        settings.canopySpread < 0.0f ||
        !std::isfinite(settings.canopyDensity) ||
        settings.canopyDensity < 0.0f) {
        throw std::invalid_argument("volume build scale is invalid");
    }

    HostSparseVolumeGrid grid;
    grid.cellResolution = settings.cellResolution;
    grid.brickResolution =
        settings.cellResolution / kVolumeBrickSize;
    grid.localBoundsMin = settings.localBoundsMin;
    grid.localBoundsMax = settings.localBoundsMax;
    grid.brickSize = kVolumeBrickSize;
    const glm::vec3 cellSize =
        (settings.localBoundsMax - settings.localBoundsMin) /
        glm::vec3(settings.cellResolution);
    grid.cellVolume = cellSize.x * cellSize.y * cellSize.z;

    const glm::ivec3 sampleResolution =
        settings.cellResolution + glm::ivec3(1);
    const std::size_t totalSamples =
        static_cast<std::size_t>(sampleResolution.x) *
        static_cast<std::size_t>(sampleResolution.y) *
        static_cast<std::size_t>(sampleResolution.z);
    std::vector<SimulationCell> current(totalSamples);
    std::vector<SimulationCell> next(totalSamples);
    std::vector<glm::vec3> baseVelocity(totalSamples);
    std::vector<float> sourceMarker(totalSamples);
    std::vector<float> markerPrediction(totalSamples);

    for (int z = 0; z < sampleResolution.z; ++z) {
        for (int y = 0; y < sampleResolution.y; ++y) {
            for (int x = 0; x < sampleResolution.x; ++x) {
                const std::size_t index =
                    sampleIndex(sampleResolution, x, y, z);
                const glm::vec3 q =
                    glm::vec3(x, y, z) /
                    glm::vec3(settings.cellResolution);
                // Advect every combustion field through the same two-scale
                // divergence-free flow.  The coarse octave establishes the
                // plume, while the finer octave folds reaction sheets before
                // semi-Lagrangian diffusion can round them into ellipsoids.
                // Because both octaves are part of the stored velocity field
                // (rather than a render-time world-space mask), density,
                // temperature, fuel, and reaction remain registered.
                const glm::vec3 curl =
                    curlNoise(q * glm::vec3(3.2f, 2.4f, 3.2f),
                              settings.seed) +
                    0.46f *
                        curlNoise(q * glm::vec3(10.8f, 7.2f, 9.6f),
                                  settings.seed + 1009);
                baseVelocity[index] =
                    baseVelocityAt(q, settings, settings.preset, curl);

                // This marker is transported by the same velocity field as
                // fuel. Reaction selects a narrow mixture-fraction band from
                // it, producing folded volumetric sheets/ribbons.
                float marker = 0.0f;
                if (settings.preset == CombustionPreset::Candle) {
                    const glm::vec2 radial(q.x - 0.5f, q.z - 0.5f);
                    marker =
                        3.2f * glm::length(radial) - 0.18f +
                        0.16f *
                            fractalNoise(
                                glm::vec3(q.x * 5.1f, q.y * 3.0f,
                                          q.z * 5.1f),
                                settings.seed + 401);
                } else if (settings.preset ==
                           CombustionPreset::WoodFire) {
                    // A depth-coherent scalar creates vertical folded
                    // mixture sheets. It is then advected by the full 3-D
                    // velocity, so the final structure bends, forks, and
                    // overlaps without degenerating into isotropic blobs.
                    marker =
                        0.64f *
                            fractalNoise(
                                glm::vec3(q.x * 4.8f, q.y * 4.7f,
                                          1.7f + q.z * 2.4f),
                                settings.seed + 401) +
                        0.36f *
                            fractalNoise(
                                glm::vec3(q.x * 12.2f, q.y * 11.3f,
                                          6.4f + q.z * 4.8f),
                                settings.seed + 607);
                } else {
                    marker =
                        0.62f *
                            fractalNoise(
                                glm::vec3(q.x * 7.2f, q.y * 4.6f,
                                          2.3f + q.z * 3.6f),
                                settings.seed + 401) +
                        0.38f *
                            fractalNoise(
                                glm::vec3(q.x * 17.5f, q.y * 11.8f,
                                          8.1f + q.z * 7.5f),
                                settings.seed + 607);
                }
                sourceMarker[index] = marker;
                current[index].marker = marker;
                current[index].velocity = baseVelocity[index];
            }
        }
    }

    const PresetParameters parameters =
        presetParameters(settings.preset);
    for (int step = 0; step < parameters.steps; ++step) {
        // Predictor for a clamped MacCormack correction of the passive
        // mixture marker. First-order semi-Lagrangian transport erased its
        // fine isosurfaces after 40+ steps even while the gas fields remained
        // plausible, which left only rounded emission lobes.
        for (int z = 0; z < sampleResolution.z; ++z) {
            for (int y = 0; y < sampleResolution.y; ++y) {
                for (int x = 0; x < sampleResolution.x; ++x) {
                    const std::size_t index =
                        sampleIndex(sampleResolution, x, y, z);
                    const glm::vec3 q =
                        glm::vec3(x, y, z) /
                        glm::vec3(settings.cellResolution);
                    const float buoyancy =
                        settings.buoyancyScale *
                        saturate(current[index].heat /
                                 parameters.maximumHeat);
                    glm::vec3 flow = baseVelocity[index];
                    flow.y +=
                        buoyancy * (0.25f + parameters.baseRise);
                    markerPrediction[index] =
                        sampleSimulation(
                            current, settings.cellResolution,
                            q - flow * parameters.dt)
                            .marker;
                }
            }
        }

        for (int z = 0; z < sampleResolution.z; ++z) {
            for (int y = 0; y < sampleResolution.y; ++y) {
                for (int x = 0; x < sampleResolution.x; ++x) {
                    const std::size_t index =
                        sampleIndex(sampleResolution, x, y, z);
                    const glm::vec3 q =
                        glm::vec3(x, y, z) /
                        glm::vec3(settings.cellResolution);
                    const float buoyancy =
                        settings.buoyancyScale *
                        saturate(current[index].heat /
                                 parameters.maximumHeat);
                    glm::vec3 flow = baseVelocity[index];
                    flow.y += buoyancy * (0.25f + parameters.baseRise);

                    const SimulationCell hotPrevious =
                        sampleSimulation(current, settings.cellResolution,
                                         q - flow * parameters.dt);
                    const glm::vec3 smokeFlow =
                        glm::vec3(flow.x, flow.y * 0.82f, flow.z);
                    const SimulationCell smokePrevious =
                        sampleSimulation(
                            current, settings.cellResolution,
                            q - smokeFlow * (parameters.dt *
                                             settings.smokeAdvection));
                    const SourceSample source = sourceAt(q, settings);

                    SimulationCell result;
                    result.velocity = flow;
                    const float markerSourceBlend =
                        settings.preset == CombustionPreset::Candle
                            ? 0.32f
                            : 0.09f;
                    const float markerReverse =
                        sampleScalar(markerPrediction,
                                     settings.cellResolution,
                                     q + flow * parameters.dt);
                    const float correctedMarker =
                        glm::clamp(
                            hotPrevious.marker +
                                0.5f *
                                    (current[index].marker -
                                     markerReverse),
                            -1.0f, 1.0f);
                    result.marker =
                        correctedMarker +
                        (sourceMarker[index] - correctedMarker) *
                            saturate(source.injection *
                                     markerSourceBlend);
                    result.marker =
                        glm::clamp(result.marker, -1.0f, 1.0f);

                    float fuel =
                        hotPrevious.fuel * parameters.fuelRetention +
                        source.injection * parameters.injectionRate;
                    const float ignition =
                        saturate(source.ignition +
                                 smoothstep(320.0f, 1050.0f,
                                            hotPrevious.heat));
                    const float markerMagnitude = std::abs(result.marker);
                    const float primaryCenter =
                        settings.preset == CombustionPreset::Wildfire
                            ? 0.11f
                            : 0.22f;
                    const float secondaryCenter =
                        settings.preset == CombustionPreset::Wildfire
                            ? 0.36f
                            : 0.49f;
                    const float primaryWidth =
                        settings.preset == CombustionPreset::Candle
                            ? 0.115f
                            : (settings.preset ==
                                       CombustionPreset::WoodFire
                                   ? 0.052f
                                   : 0.065f);
                    const float primarySheet =
                        std::exp(-square(
                            (markerMagnitude - primaryCenter) /
                            primaryWidth));
                    const float secondarySheet =
                        std::exp(-square(
                            (markerMagnitude - secondaryCenter) /
                            (settings.preset ==
                                     CombustionPreset::Wildfire
                                 ? 0.052f
                                 : 0.040f)));
                    const float foldedSheet =
                        std::max(primarySheet, 0.68f * secondarySheet);
                    const float burnMixture =
                        settings.preset == CombustionPreset::Candle
                            ? 0.55f + 0.45f * primarySheet
                            : (settings.preset ==
                                       CombustionPreset::Wildfire
                                   ? 0.20f + 0.80f * foldedSheet
                                   : 0.22f + 0.78f * foldedSheet);
                    const float reactionPotential =
                        fuel * burnMixture *
                        (0.14f + 0.86f * ignition);
                    const float burnedFuel =
                        std::min(fuel, reactionPotential *
                                          parameters.burnRate);
                    fuel -= burnedFuel;

                    result.fuel = std::max(0.0f, fuel);
                    const float reactionSheet =
                        settings.preset == CombustionPreset::Candle
                            ? 0.30f + 0.70f * primarySheet
                            : ((settings.preset ==
                                        CombustionPreset::WoodFire
                                    ? 0.08f
                                    : 0.10f) +
                               (settings.preset ==
                                        CombustionPreset::WoodFire
                                    ? 0.92f
                                    : 0.90f) *
                                   foldedSheet);
                    result.reaction =
                        saturate(std::max(
                            hotPrevious.reaction *
                                parameters.reactionRetention,
                            burnedFuel * parameters.reactionGain *
                                reactionSheet));
                    result.heat = std::min(
                        parameters.maximumHeat,
                        std::max(
                            0.0f,
                            hotPrevious.heat *
                                    parameters.heatRetention +
                                burnedFuel * parameters.heatYield +
                                source.ignition * parameters.sourceHeat));
                    result.density = saturate(
                        smokePrevious.density *
                                parameters.densityRetention +
                            burnedFuel * parameters.gasYield);
                    result.soot = saturate(
                        smokePrevious.soot *
                                parameters.sootRetention +
                            burnedFuel * parameters.sootYield);
                    next[index] = result;
                }
            }
        }
        current.swap(next);
    }

    const std::size_t pageCount =
        static_cast<std::size_t>(grid.brickResolution.x) *
        static_cast<std::size_t>(grid.brickResolution.y) *
        static_cast<std::size_t>(grid.brickResolution.z);
    grid.pageTable.assign(pageCount, -1);

    for (int brickZ = 0; brickZ < grid.brickResolution.z; ++brickZ) {
        for (int brickY = 0; brickY < grid.brickResolution.y; ++brickY) {
            for (int brickX = 0; brickX < grid.brickResolution.x;
                 ++brickX) {
                std::array<glm::vec4, kVolumeBrickSampleCount>
                    combustion{};
                std::array<glm::vec4, kVolumeBrickSampleCount>
                    thermalFlow{};
                VolumeBrickMeta meta;
                meta.brickX = brickX;
                meta.brickY = brickY;
                meta.brickZ = brickZ;

                float activity = 0.0f;
                for (int sampleZ = 0;
                     sampleZ < kVolumeBrickSampleSize; ++sampleZ) {
                    for (int sampleY = 0;
                         sampleY < kVolumeBrickSampleSize; ++sampleY) {
                        for (int sampleX = 0;
                             sampleX < kVolumeBrickSampleSize;
                             ++sampleX) {
                            const int gridX =
                                brickX * kVolumeBrickSize + sampleX;
                            const int gridY =
                                brickY * kVolumeBrickSize + sampleY;
                            const int gridZ =
                                brickZ * kVolumeBrickSize + sampleZ;
                            const std::size_t denseIndex =
                                sampleIndex(sampleResolution, gridX,
                                            gridY, gridZ);
                            const glm::vec3 q =
                                glm::vec3(gridX, gridY, gridZ) /
                                glm::vec3(settings.cellResolution);
                            const SourceSample source =
                                sourceAt(q, settings);
                            const SimulationCell &cell =
                                current[denseIndex];

                            const float sootCanopy =
                                woodSootCanopyAt(q, settings) *
                                settings.canopyDensity;
                            const float density =
                                std::max(
                                    std::max(0.0f, cell.density) *
                                        settings.densityScale,
                                    0.62f * sootCanopy);
                            float soot =
                                std::max(
                                    std::max(0.0f, cell.soot) *
                                        settings.sootScale,
                                    sootCanopy);
                            const float fuel =
                                std::max(cell.fuel, source.solidFuel) *
                                settings.fuelScale;
                            float reaction =
                                std::max(0.0f, cell.reaction);
                            float wildfireSootCarrierHeat = 0.0f;
                            float wildfireSheetHeatStrength = 0.0f;
                            float wildfireCoreHeatStrength = 0.0f;
                            float wildfireBodyHeatStrength = 0.0f;
                            if (settings.preset !=
                                CombustionPreset::Candle) {
                                // Store the hot reaction front as narrow
                                // advected mixture-fraction sheets. Density,
                                // soot, and temperature remain broader, so
                                // these become internal HDR ribbons rather
                                // than isolated geometric flame primitives.
                                const float markerMagnitude =
                                    std::abs(cell.marker);
                                const bool wildfire =
                                    settings.preset ==
                                    CombustionPreset::Wildfire;
                                const float primaryCenter =
                                    wildfire ? 0.11f : 0.22f;
                                const float secondaryCenter =
                                    wildfire ? 0.36f : 0.49f;
                                const float primaryRibbon =
                                    std::exp(-square(
                                        (markerMagnitude -
                                         primaryCenter) /
                                        (wildfire ? 0.040f :
                                                    0.038f)));
                                const float secondaryRibbon =
                                    std::exp(-square(
                                        (markerMagnitude -
                                         secondaryCenter) /
                                        (wildfire ? 0.032f :
                                                    0.030f)));
                                const float ribbon =
                                    std::max(primaryRibbon,
                                             0.72f * secondaryRibbon);

                                const auto reactionAt =
                                    [&](int offsetX, int offsetY,
                                        int offsetZ) {
                                        const int nx = std::clamp(
                                            gridX + offsetX, 0,
                                            settings.cellResolution.x);
                                        const int ny = std::clamp(
                                            gridY + offsetY, 0,
                                            settings.cellResolution.y);
                                        const int nz = std::clamp(
                                            gridZ + offsetZ, 0,
                                            settings.cellResolution.z);
                                        return current[sampleIndex(
                                                           sampleResolution,
                                                           nx, ny, nz)]
                                            .reaction;
                                    };
                                const auto markerAt =
                                    [&](int offsetX, int offsetY,
                                        int offsetZ) {
                                        const int nx = std::clamp(
                                            gridX + offsetX, 0,
                                            settings.cellResolution.x);
                                        const int ny = std::clamp(
                                            gridY + offsetY, 0,
                                            settings.cellResolution.y);
                                        const int nz = std::clamp(
                                            gridZ + offsetZ, 0,
                                            settings.cellResolution.z);
                                        return current[sampleIndex(
                                                           sampleResolution,
                                                           nx, ny, nz)]
                                            .marker;
                                    };
                                const auto heatAt =
                                    [&](int offsetX, int offsetY,
                                        int offsetZ) {
                                        const int nx = std::clamp(
                                            gridX + offsetX, 0,
                                            settings.cellResolution.x);
                                        const int ny = std::clamp(
                                            gridY + offsetY, 0,
                                            settings.cellResolution.y);
                                        const int nz = std::clamp(
                                            gridZ + offsetZ, 0,
                                            settings.cellResolution.z);
                                        return current[sampleIndex(
                                                           sampleResolution,
                                                           nx, ny, nz)]
                                            .heat;
                                    };
                                const glm::vec3 reactionGradient(
                                    0.5f *
                                        (reactionAt(1, 0, 0) -
                                         reactionAt(-1, 0, 0)),
                                    0.5f *
                                        (reactionAt(0, 1, 0) -
                                         reactionAt(0, -1, 0)),
                                    0.5f *
                                        (reactionAt(0, 0, 1) -
                                         reactionAt(0, 0, -1)));
                                const glm::vec3 markerGradient(
                                    0.5f *
                                        (markerAt(1, 0, 0) -
                                         markerAt(-1, 0, 0)),
                                    0.5f *
                                        (markerAt(0, 1, 0) -
                                         markerAt(0, -1, 0)),
                                    0.5f *
                                        (markerAt(0, 0, 1) -
                                         markerAt(0, 0, -1)));
                                const glm::vec3 heatGradient(
                                    0.5f *
                                        (heatAt(1, 0, 0) -
                                         heatAt(-1, 0, 0)),
                                    0.5f *
                                        (heatAt(0, 1, 0) -
                                         heatAt(0, -1, 0)),
                                    0.5f *
                                        (heatAt(0, 0, 1) -
                                         heatAt(0, 0, -1)));
                                const float reactionEdge =
                                    smoothstep(
                                        0.018f, 0.16f,
                                        glm::length(reactionGradient));
                                const float broadMixSheet =
                                    std::max(
                                        std::exp(-square(
                                            (markerMagnitude -
                                             primaryCenter) /
                                            (wildfire ? 0.14f :
                                                        0.11f))),
                                        0.58f *
                                            std::exp(-square(
                                                (markerMagnitude -
                                                 secondaryCenter) /
                                                (wildfire ? 0.11f :
                                                            0.085f))));
                                const float foldedFront =
                                    std::pow(reactionEdge, 1.35f) *
                                    (0.05f +
                                     0.95f *
                                         std::pow(broadMixSheet, 1.45f));

                                // Preserve a dim, connected hot carrier near
                                // the actual fuel while reserving HDR gas
                                // emission for thin transported isosurfaces.
                                // The previous additive floor was only 2.5%,
                                // but Planck amplification at 2500 K made its
                                // broad skirts dominate the image as smooth
                                // glowing blobs.
                                const float rootCarrier =
                                    source.ignition *
                                    std::exp(-q.y /
                                             (wildfire ? 0.075f :
                                                         0.055f));
                                const float zeroCrossing =
                                    std::exp(-square(
                                        cell.marker /
                                        (wildfire ? 0.050f :
                                                    0.045f))) *
                                    smoothstep(
                                        0.004f, 0.055f,
                                        glm::length(markerGradient));
                                const float sheet =
                                    std::max(
                                        zeroCrossing,
                                        std::max(
                                            0.38f *
                                                std::pow(ribbon, 1.60f),
                                            0.28f *
                                                std::pow(foldedFront,
                                                         1.15f)));
                                const float hotGas =
                                    smoothstep(620.0f, 1720.0f,
                                               cell.heat);
                                const float freshGas =
                                    saturate(1.45f * cell.density +
                                             0.35f * cell.fuel);
                                const float simulatedReaction =
                                    std::pow(saturate(reaction), 0.82f);
                                const float transportedCarrier =
                                    hotGas *
                                    (0.10f + 0.90f * freshGas);
                                const float transportedSheet =
                                    std::max(simulatedReaction,
                                             transportedCarrier) *
                                    sheet;
                                const float thermalFront =
                                    hotGas *
                                    smoothstep(
                                        8.0f, 95.0f,
                                        glm::length(heatGradient)) *
                                    (0.18f + 0.82f * freshGas);
                                const float hotRoot =
                                    (wildfire ? 0.12f : 0.085f) *
                                    rootCarrier *
                                    (0.20f +
                                     0.80f *
                                         smoothstep(120.0f, 850.0f,
                                                    cell.heat));
                                reaction =
                                    std::max(
                                        std::max(transportedSheet,
                                                 0.18f *
                                                     thermalFront *
                                                     sheet),
                                        hotRoot);
                                reaction = saturate(
                                    reaction *
                                    (wildfire ? 3.4f : 4.0f));
                                // Keep the broad transported carrier below
                                // the hottest reconstructed reaction sheets.
                                // This preserves orange body/detail instead
                                // of allowing a large saturated yellow lobe.
                                reaction =
                                    std::min(reaction,
                                             wildfire ? 0.004f : 0.003f);

                                // Reconstruct fuel-attached flame bodies from
                                // the stored, coherently advected velocity and
                                // mixture marker. The broad term keeps roots
                                // attached to actual fuel while the narrow
                                // term supplies hotter folded structures
                                // inside it. Both are baked into the grid;
                                // neither is evaluated by the renderer.
                                glm::vec3 horizontalFlow = cell.velocity;
                                horizontalFlow.y = 0.0f;
                                const float rootTrace =
                                    wildfire
                                        ? (0.28f * q.y +
                                           0.55f * q.y * q.y)
                                        : (0.26f * q.y +
                                           0.20f * q.y * q.y);
                                glm::vec3 rootProbe =
                                    q - horizontalFlow * rootTrace;
                                float woodCoverage = 0.0f;
                                float woodRootUnionCoverage = 0.0f;
                                float woodHeightScale = 1.0f;
                                float rootLevel = 0.0f;
                                if (wildfire) {
                                    rootLevel =
                                        0.020f +
                                        0.040f *
                                            smoothstep(
                                                -0.72f, 0.70f,
                                                fractalNoise(
                                                    glm::vec3(
                                                        rootProbe.x * 9.2f,
                                                        5.1f,
                                                        rootProbe.z * 8.4f),
                                                    settings.seed + 1777));
                                } else {
                                    glm::vec3 woodSourceProbe = rootProbe;
                                    woodSourceProbe.x =
                                        (rootProbe.x - 0.5f) *
                                            settings.sourceCompactness +
                                        0.5f;
                                    woodSourceProbe.z =
                                        (rootProbe.z - 0.5f) *
                                            settings.sourceCompactness +
                                        0.5f;
                                    float sumWeight = 0.0f;
                                    float maxWeight = 0.0f;
                                    float weightedHeight = 0.0f;
                                    float weightedHeightScale = 0.0f;
                                    float unionWeightSum = 0.0f;
                                    const float footprintExpansion =
                                        1.0f +
                                        0.80f *
                                            smoothstep(
                                                0.18f, 0.42f, q.y);
                                    for (const WoodBurner &burner :
                                         kWoodBurners) {
                                        const float dx =
                                            (woodSourceProbe.x -
                                             burner.center.x) /
                                            (burner.radius.x *
                                             footprintExpansion);
                                        const float dz =
                                            (woodSourceProbe.z -
                                             burner.center.z) /
                                            (burner.radius.z *
                                             footprintExpansion);
                                        const float footprint =
                                            burner.strength *
                                            std::exp(-(dx * dx + dz * dz));
                                        const float unionDx =
                                            (woodSourceProbe.x -
                                             burner.center.x) /
                                            (burner.radius.x * 2.00f);
                                        const float unionDz =
                                            (woodSourceProbe.z -
                                             burner.center.z) /
                                            (burner.radius.z * 1.80f);
                                        unionWeightSum +=
                                            burner.strength *
                                            std::exp(
                                                -(unionDx * unionDx +
                                                  unionDz * unionDz));
                                        const float weight =
                                            smoothstep(0.06f, 0.48f,
                                                       footprint);
                                        sumWeight += weight;
                                        maxWeight =
                                            std::max(maxWeight, weight);
                                        weightedHeight +=
                                            weight * burner.center.y;
                                        weightedHeightScale +=
                                            weight *
                                            burner.heightScale;
                                    }
                                    rootLevel =
                                        sumWeight > 1.0e-5f
                                            ? weightedHeight / sumWeight
                                            : 0.075f;
                                    woodHeightScale =
                                        sumWeight > 1.0e-5f
                                            ? weightedHeightScale /
                                                  sumWeight
                                            : 1.0f;
                                    // Preserve discrete burner identities but
                                    // retain a weaker bridge where two
                                    // adjacent log-gap pockets overlap.
                                    const float secondaryWeight =
                                        std::min(
                                            1.0f,
                                            std::max(
                                                0.0f,
                                                sumWeight - maxWeight));
                                    woodCoverage =
                                        saturate(0.68f * maxWeight +
                                                 0.05f *
                                                     secondaryWeight);
                                    woodRootUnionCoverage =
                                        1.0f -
                                        std::exp(
                                            -0.50f *
                                            unionWeightSum);
                                }
                                rootProbe.y = rootLevel;
                                const SourceSample rootSource =
                                    sourceAt(rootProbe, settings);
                                // Analytic source kernels have infinitesimal
                                // Gaussian tails. Compact their support before
                                // reconstructing transported tongues so those
                                // tails cannot activate every sparse brick or
                                // dilute the emission hierarchy.
                                const float rootIgnition =
                                    wildfire
                                        ? smoothstep(
                                              0.020f, 0.28f,
                                              rootSource.ignition)
                                        : smoothstep(
                                              0.18f, 0.78f,
                                              woodCoverage);
                                const float markerPattern =
                                    smoothstep(
                                        wildfire ? 0.18f : 0.14f,
                                        wildfire ? 0.82f : 0.86f,
                                        0.5f + 0.5f * cell.marker);
                                const float heightPattern =
                                    0.5f +
                                    0.5f *
                                        fractalNoise(
                                            glm::vec3(
                                                rootProbe.x *
                                                    (wildfire ? 11.0f :
                                                                6.5f),
                                                3.4f,
                                                rootProbe.z *
                                                    (wildfire ? 9.0f :
                                                                5.8f)),
                                            settings.seed + 2017);
                                const float irregularHeight =
                                    wildfire
                                        ? (0.052f +
                                           0.150f *
                                               std::pow(markerPattern,
                                                        1.10f))
                                        : woodHeightScale *
                                              (0.14f +
                                               0.29f *
                                                   std::pow(
                                                       markerPattern,
                                                       0.90f) +
                                               0.24f *
                                                   smoothstep(
                                                       0.68f, 0.93f,
                                                       heightPattern));
                                const float heightAboveRoot =
                                    std::max(0.0f, q.y - rootLevel);
                                const float rootGate =
                                    smoothstep(-0.006f, 0.018f,
                                               q.y - rootLevel);
                                const float rootUnionNoise =
                                    0.5f +
                                    0.5f *
                                        fractalNoise(
                                            glm::vec3(
                                                rootProbe.x * 8.6f,
                                                q.y * 5.2f,
                                                rootProbe.z * 7.8f),
                                            settings.seed + 2297);
                                const float rootUnionHeight =
                                    0.045f +
                                    0.055f *
                                        smoothstep(
                                            0.18f, 0.86f,
                                            rootUnionNoise);
                                const float rootUnionFalloff =
                                    1.0f -
                                    smoothstep(
                                        0.48f, 1.05f,
                                        heightAboveRoot /
                                            std::max(
                                                rootUnionHeight,
                                                1.0e-4f));
                                const float heightRatio =
                                    heightAboveRoot /
                                    std::max(irregularHeight, 1.0e-4f);
                                const float reconstructedSupport =
                                    wildfire
                                        ? rootIgnition
                                        : std::pow(
                                              rootIgnition,
                                              1.0f +
                                                  0.55f *
                                                      saturate(
                                                          heightRatio));
                                const float bodyTip =
                                    1.0f -
                                    smoothstep(0.72f, 1.05f,
                                               heightRatio);
                                const float ribbonTip =
                                    1.0f -
                                    smoothstep(0.82f, 1.20f,
                                               heightRatio);
                                const float bodyFalloff =
                                    std::exp(
                                        -std::pow(heightRatio,
                                                  wildfire ? 1.75f :
                                                             1.65f)) *
                                    bodyTip;
                                const float ribbonFalloff =
                                    std::exp(
                                        -std::pow(
                                            heightAboveRoot /
                                                std::max(
                                                    irregularHeight *
                                                        (wildfire ? 1.16f :
                                                                    1.08f),
                                                    1.0e-4f),
                                            wildfire ? 1.52f : 1.48f));
                                const float boundedRibbonFalloff =
                                    ribbonFalloff * ribbonTip * rootGate;
                                // A third, finer scalar is evaluated at the
                                // velocity-backtraced probe, so it is coherent
                                // with the stored flow rather than a vertical
                                // world-space texture. Two narrow isovalues
                                // create irregular folded highlight streams;
                                // their variable finite height supplies many
                                // short wildfire tongues and smaller flames
                                // between the wood-fire bursts.
                                const float detailTrace =
                                    wildfire
                                        ? rootTrace
                                        : (0.38f * q.y +
                                           0.32f * q.y * q.y);
                                const glm::vec3 detailProbe =
                                    q - horizontalFlow * detailTrace;
                                const float advectedDetail =
                                    0.5f +
                                    0.5f *
                                        fractalNoise(
                                            detailProbe *
                                                (wildfire
                                                     ? glm::vec3(
                                                           18.5f, 6.2f,
                                                           15.7f)
                                                     : glm::vec3(
                                                           7.5f, 3.8f,
                                                            6.5f)),
                                            settings.seed + 1597);
                                const float detailSegmentation =
                                    0.5f +
                                    0.5f *
                                        fractalNoise(
                                            detailProbe *
                                                    (wildfire
                                                         ? glm::vec3(
                                                               10.7f,
                                                               7.8f,
                                                               12.3f)
                                                         : glm::vec3(
                                                               5.2f,
                                                               4.6f,
                                                               5.7f)) +
                                                glm::vec3(
                                                    2.3f, 5.9f,
                                                    8.1f),
                                            settings.seed + 3811);
                                const float detailHeight =
                                    wildfire
                                        ? (0.040f +
                                           0.070f *
                                               smoothstep(
                                                   0.20f, 0.82f,
                                                   advectedDetail) +
                                           0.135f *
                                               smoothstep(
                                                   0.86f, 0.97f,
                                                   detailSegmentation))
                                        : woodHeightScale *
                                              (0.11f +
                                               0.28f *
                                                   smoothstep(
                                                       0.10f, 0.90f,
                                                       advectedDetail) +
                                               0.18f *
                                                   smoothstep(
                                                       0.78f, 0.96f,
                                                       heightPattern));
                                const float detailHeightRatio =
                                    heightAboveRoot /
                                    std::max(detailHeight, 1.0e-4f);
                                const float detailTip =
                                    1.0f -
                                    smoothstep(0.70f, 1.10f,
                                               detailHeightRatio);
                                const float detailFalloff =
                                    std::exp(
                                        -std::pow(detailHeightRatio,
                                                  wildfire ? 1.62f :
                                                             1.52f)) *
                                    detailTip * rootGate;
                                const float ribbonWidthSignal =
                                    0.5f +
                                    0.5f *
                                        fractalNoise(
                                            detailProbe *
                                                (wildfire
                                                     ? glm::vec3(
                                                           7.5f, 4.1f,
                                                           6.8f)
                                                     : glm::vec3(
                                                           4.4f, 3.2f,
                                                           5.1f)),
                                            settings.seed + 2699);
                                const float ribbonOpening =
                                    wildfire
                                        ? 1.0f
                                        : smoothstep(
                                              0.26f, 0.50f,
                                              ribbonWidthSignal);
                                float detailRidge = 0.0f;
                                float wildfireSheetPatch = 0.0f;
                                float wildfireCorePatch = 0.0f;
                                float wildfireBodyPatch = 0.0f;
                                float wildfireDetailSupport = 0.0f;
                                if (wildfire) {
                                    // Wildfire flames are open surfaces
                                    // attached to the continuous procedural
                                    // fuel front.  The signed front distance
                                    // is evaluated after the velocity
                                    // backtrace, so wind and curl turbulence
                                    // bend the full 3-D sheets without
                                    // intersecting two closed noise contours
                                    // into beads.
                                    const float signedFrontDistance =
                                        rootProbe.z -
                                        wildfireFrontZAt(
                                            rootProbe.x,
                                            settings.seed);
                                    const float rootHeightNoiseA =
                                        0.5f +
                                        0.5f *
                                            fractalNoise(
                                                glm::vec3(
                                                    rootProbe.x * 16.2f,
                                                    3.2f,
                                                    rootProbe.z * 11.4f),
                                                settings.seed + 4073);
                                    const float rootHeightNoiseB =
                                        0.5f +
                                        0.5f *
                                            fractalNoise(
                                                glm::vec3(
                                                    rootProbe.x * 4.5f,
                                                    7.1f,
                                                    rootProbe.z * 3.8f),
                                                settings.seed + 4129);
                                    const float rareFlareBase =
                                        smoothstep(
                                            0.62f, 0.80f,
                                            rootHeightNoiseB);
                                    const float rareFlare =
                                        rareFlareBase *
                                        (0.24f +
                                         0.76f *
                                             smoothstep(
                                                 0.34f, 0.74f,
                                                 rootHeightNoiseA));
                                    const float sheetHeight =
                                        0.105f +
                                        0.085f *
                                            std::pow(
                                                rootHeightNoiseA,
                                                1.30f) +
                                        0.090f *
                                            std::pow(
                                                rareFlare,
                                                1.60f);
                                    const float h =
                                        heightAboveRoot /
                                        std::max(
                                            sheetHeight,
                                            1.0e-4f);
                                    const float sheetY =
                                        rootGate *
                                        std::exp(
                                            -std::pow(h, 1.42f)) *
                                        (1.0f -
                                             smoothstep(
                                                 0.84f, 1.12f,
                                                 h));
                                    const float independentHeightB =
                                        saturate(
                                            0.72f * heightPattern +
                                            0.28f *
                                                (1.0f -
                                                 rootHeightNoiseA));
                                    const float independentHeightC =
                                        saturate(
                                            0.55f *
                                                (1.0f -
                                                 rootHeightNoiseA) +
                                            0.45f *
                                                (1.0f -
                                                 heightPattern));
                                    const float rareFlareB =
                                        smoothstep(
                                            0.68f, 0.88f,
                                            0.55f *
                                                    rootHeightNoiseB +
                                                0.45f *
                                                    heightPattern);
                                    const float rareFlareC =
                                        smoothstep(
                                            0.70f, 0.90f,
                                            0.50f *
                                                    rootHeightNoiseB +
                                                0.50f *
                                                    (1.0f -
                                                     heightPattern));
                                    const float sheetHeightB =
                                        0.090f +
                                        0.085f *
                                            std::pow(
                                                independentHeightB,
                                                1.20f) +
                                        0.075f *
                                            std::pow(
                                                rareFlareB, 1.50f);
                                    const float sheetHeightC =
                                        0.080f +
                                        0.075f *
                                            std::pow(
                                                independentHeightC,
                                                1.25f) +
                                        0.055f *
                                            std::pow(
                                                rareFlareC, 1.50f);
                                    const float hB =
                                        heightAboveRoot /
                                        std::max(
                                            sheetHeightB,
                                            1.0e-4f);
                                    const float hC =
                                        heightAboveRoot /
                                        std::max(
                                            sheetHeightC,
                                            1.0e-4f);
                                    const float sheetYB =
                                        rootGate *
                                        std::exp(
                                            -std::pow(hB, 1.36f)) *
                                        (1.0f -
                                         smoothstep(
                                             0.80f, 1.10f,
                                             hB));
                                    const float sheetYC =
                                        rootGate *
                                        std::exp(
                                            -std::pow(hC, 1.30f)) *
                                        (1.0f -
                                         smoothstep(
                                             0.76f, 1.08f,
                                             hC));
                                    const glm::vec3 foldProbeA(
                                        detailProbe.x +
                                            0.52f * detailProbe.y,
                                        detailProbe.y +
                                            0.34f * detailProbe.x,
                                        detailProbe.z +
                                            0.21f * detailProbe.y);
                                    const glm::vec3 foldProbeB(
                                        detailProbe.x -
                                            0.39f * detailProbe.y,
                                        detailProbe.y +
                                            0.27f * detailProbe.z,
                                        detailProbe.z -
                                            0.24f * detailProbe.x);
                                    const glm::vec3 foldProbeC(
                                        detailProbe.x +
                                            0.31f * detailProbe.z,
                                        detailProbe.y -
                                            0.42f * detailProbe.x,
                                        detailProbe.z +
                                            0.29f * detailProbe.y);
                                    const float foldNoiseA =
                                        fractalNoise(
                                            foldProbeA *
                                                    glm::vec3(
                                                        12.8f, 13.4f,
                                                        11.8f) +
                                                glm::vec3(
                                                    1.7f, 8.3f, 3.1f),
                                            settings.seed + 4211);
                                    const float foldNoiseB =
                                        fractalNoise(
                                            foldProbeB *
                                                    glm::vec3(
                                                        16.1f, 11.6f,
                                                        14.1f) +
                                                glm::vec3(
                                                    7.4f, 2.6f, 9.1f),
                                            settings.seed + 4271);
                                    const float foldNoiseC =
                                        fractalNoise(
                                            foldProbeC *
                                                    glm::vec3(
                                                        19.3f, 15.2f,
                                                        16.7f) +
                                                glm::vec3(
                                                    4.9f, 6.2f, 1.4f),
                                            settings.seed + 4337);
                                    const float sheetWidth =
                                        0.0190f +
                                        0.0080f *
                                            ribbonWidthSignal;
                                    const float foldGrowth =
                                        smoothstep(
                                            0.02f, 0.78f,
                                            h);
                                    const float signedSheetA =
                                        signedFrontDistance +
                                        (0.010f +
                                         0.075f * foldGrowth) *
                                            foldNoiseA +
                                        0.014f * h *
                                            foldNoiseB;
                                    const float signedSheetB =
                                        signedFrontDistance -
                                        0.023f +
                                        (0.009f +
                                         0.067f * foldGrowth) *
                                            foldNoiseB -
                                        0.012f * h *
                                            foldNoiseC;
                                    const float signedSheetC =
                                        signedFrontDistance +
                                        0.027f +
                                        (0.008f +
                                         0.062f * foldGrowth) *
                                            foldNoiseC +
                                        0.010f * h *
                                            foldNoiseA;
                                    const float openSheetA =
                                        std::exp(
                                            -square(
                                                signedSheetA /
                                                sheetWidth));
                                    const float openSheetB =
                                        0.74f *
                                        std::exp(
                                            -square(
                                                signedSheetB /
                                                (0.90f *
                                                 sheetWidth)));
                                    const float openSheetC =
                                        0.58f *
                                        std::exp(
                                            -square(
                                                signedSheetC /
                                                (0.82f *
                                                 sheetWidth)));
                                    // Erode only the exposed crowns with
                                    // independent, coherently advected 3-D
                                    // signals. Roots remain uncut, while the
                                    // three depth-offset sheets split and
                                    // overlap differently above mid-height.
                                    // Cross-using the existing fold fields
                                    // avoids a second periodic height mask
                                    // and keeps this a single stored volume,
                                    // not a set of authored tongues.
                                    const float crownSignalA =
                                        0.36f *
                                            (0.5f +
                                             0.5f * foldNoiseB) +
                                        0.28f *
                                            advectedDetail +
                                        0.36f *
                                            markerPattern;
                                    const float crownSignalB =
                                        0.42f *
                                            (0.5f +
                                             0.5f * foldNoiseC) +
                                        0.34f *
                                            detailSegmentation +
                                        0.24f *
                                            advectedDetail;
                                    const float crownSignalC =
                                        0.40f *
                                            (0.5f +
                                             0.5f * foldNoiseA) +
                                        0.34f *
                                            (1.0f -
                                             advectedDetail) +
                                        0.26f *
                                            markerPattern;
                                    const float crownGateA =
                                        0.025f +
                                        0.975f *
                                            smoothstep(
                                                0.34f, 0.66f,
                                                crownSignalA);
                                    const float crownGateB =
                                        0.035f +
                                        0.965f *
                                            smoothstep(
                                                0.32f, 0.68f,
                                                crownSignalB);
                                    const float crownGateC =
                                        0.045f +
                                        0.955f *
                                            smoothstep(
                                                0.35f, 0.65f,
                                                crownSignalC);
                                    const float crownSurvivalA =
                                        1.0f -
                                        smoothstep(
                                            0.18f, 0.82f, h) *
                                            (1.0f - crownGateA);
                                    const float crownSurvivalB =
                                        1.0f -
                                        smoothstep(
                                            0.16f, 0.80f, hB) *
                                            (1.0f - crownGateB);
                                    const float crownSurvivalC =
                                        1.0f -
                                        smoothstep(
                                            0.15f, 0.78f, hC) *
                                            (1.0f - crownGateC);
                                    const float sheetEnvelope =
                                        std::max(
                                            sheetY *
                                                crownSurvivalA *
                                                openSheetA,
                                            std::max(
                                                sheetYB *
                                                    crownSurvivalB *
                                                    openSheetB,
                                                sheetYC *
                                                    crownSurvivalC *
                                                    openSheetC));
                                    // Irregular pale sub-sheet highlights are
                                    // contours of coherently advected scalar
                                    // fields, not a fixed set of authored
                                    // roots. Intersecting each broad contour
                                    // with a different folded open sheet
                                    // yields narrow 3-D ribbon regions whose
                                    // spacing, split points, and lifetime are
                                    // entirely field-driven. Because the
                                    // orange sheet remains the primary
                                    // reaction source, partial ridge contours
                                    // cannot read as standalone flames.
                                    const float ridgeA =
                                        std::exp(
                                            -square(
                                                (advectedDetail -
                                                 (0.42f +
                                                  0.08f *
                                                      foldNoiseC)) /
                                                0.060f));
                                    const float ridgeB =
                                        std::exp(
                                            -square(
                                                (detailSegmentation -
                                                 (0.58f +
                                                  0.10f *
                                                      foldNoiseA)) /
                                                0.065f));
                                    const float ridgeC =
                                        std::exp(
                                            -square(
                                                ((0.5f +
                                                  0.5f *
                                                      foldNoiseB) -
                                                 (0.52f +
                                                  0.07f *
                                                      foldNoiseC)) /
                                                0.075f));
                                    const float hotRidgeEnvelope =
                                        std::max(
                                            sheetY *
                                                crownSurvivalA *
                                                std::pow(
                                                    openSheetA, 1.45f) *
                                                ridgeA,
                                            std::max(
                                                0.72f *
                                                    sheetYB *
                                                    crownSurvivalB *
                                                    std::pow(
                                                        openSheetB /
                                                            0.74f,
                                                        1.35f) *
                                                    ridgeB,
                                                0.52f *
                                                    sheetYC *
                                                    crownSurvivalC *
                                                    std::pow(
                                                        openSheetC /
                                                            0.58f,
                                                        1.30f) *
                                                    ridgeC));
                                    // The backtraced fuel support is exact at
                                    // the root. Above it, the reconstructed
                                    // flame sheet may fold sideways beyond
                                    // the narrow ground source band. Expand
                                    // support continuously with fold growth
                                    // so those advected upper folds are not
                                    // clipped back into smooth source lobes.
                                    const float sheetAdvectionSupport =
                                        saturate(
                                            reconstructedSupport +
                                            0.88f * foldGrowth *
                                                (1.0f -
                                                 reconstructedSupport));
                                    // Broad advected detail changes sheet
                                    // power and thickness, but retains a
                                    // connected support floor.
                                    wildfireDetailSupport =
                                        0.18f +
                                        0.82f *
                                            smoothstep(
                                                0.14f, 0.86f,
                                                advectedDetail);
                                    const float patchMod =
                                        0.28f +
                                        0.72f *
                                            smoothstep(
                                                0.16f, 0.82f,
                                                detailSegmentation);
                                    const float verticalBreakupSignal =
                                        0.5f +
                                        0.5f *
                                            fractalNoise(
                                                detailProbe *
                                                        glm::vec3(
                                                            9.4f, 13.7f,
                                                            11.2f) +
                                                    glm::vec3(
                                                        6.8f, 1.9f,
                                                        4.3f),
                                                settings.seed + 4421);
                                    const float verticalBreakup =
                                        0.24f +
                                        0.76f *
                                            smoothstep(
                                                0.24f, 0.78f,
                                                verticalBreakupSignal);
                                    // Preserve a visible orange connection
                                    // along the continuous open sheet. The
                                    // previous direct product of three masks
                                    // fell to roughly 1.2%, so thin-sheet
                                    // view-angle amplification exposed only
                                    // isolated tangent folds even though the
                                    // underlying sheet was connected.
                                    const float structuralSheet =
                                        saturate(
                                            sheetAdvectionSupport *
                                            sheetEnvelope);
                                    const float breakupProduct =
                                        wildfireDetailSupport *
                                        patchMod *
                                        verticalBreakup;
                                    const float sheetGain =
                                        0.22f +
                                        0.78f *
                                            breakupProduct;
                                    wildfireSheetPatch =
                                        saturate(
                                            structuralSheet *
                                            sheetGain);
                                    wildfireCorePatch =
                                        saturate(
                                            sheetAdvectionSupport *
                                            hotRidgeEnvelope);
                                    // A dimmer, fuller gas body keeps the
                                    // roots continuous and orange.  It is
                                    // deliberately shorter than the hot open
                                    // sheets, whose folds carry pale-yellow
                                    // highlights.
                                    const float bodyHeight =
                                        0.028f +
                                        0.038f *
                                            (0.35f +
                                             0.65f *
                                                 rootHeightNoiseA) +
                                        0.008f * rareFlare;
                                    const float bodyH =
                                        heightAboveRoot /
                                        std::max(
                                            bodyHeight,
                                            1.0e-4f);
                                    const float bodyY =
                                        rootGate *
                                        std::exp(
                                            -std::pow(
                                                bodyH, 1.65f)) *
                                        (1.0f -
                                         smoothstep(
                                             0.80f, 1.08f,
                                             bodyH));
                                    wildfireBodyPatch =
                                        saturate(
                                            reconstructedSupport *
                                            bodyY *
                                            (0.30f +
                                             0.70f *
                                                 std::sqrt(
                                                     saturate(
                                                         rootSource.solidFuel))) *
                                            (0.62f +
                                             0.38f *
                                                 wildfireDetailSupport));
                                } else {
                                    // Three independently warped open sheets,
                                    // rather than parallel copies or closed
                                    // noise isosurfaces, form the wood-fire
                                    // HDR cores. Each is evaluated after the
                                    // nonlinear flow backtrace and baked into
                                    // the sparse reaction field.
                                    const float warpA =
                                        fractalNoise(
                                            detailProbe *
                                                glm::vec3(
                                                    4.8f, 5.1f, 5.5f),
                                            settings.seed + 3181);
                                    const float warpB =
                                        fractalNoise(
                                            detailProbe *
                                                    glm::vec3(
                                                        5.7f, 4.3f, 4.9f) +
                                                glm::vec3(
                                                    3.1f, 7.7f, 1.9f),
                                            settings.seed + 3253);
                                    const float warpC =
                                        fractalNoise(
                                            detailProbe *
                                                    glm::vec3(
                                                        4.2f, 5.8f, 6.1f) +
                                                glm::vec3(
                                                    8.3f, 2.4f, 5.6f),
                                            settings.seed + 3347);
                                    const float sheetA =
                                        detailProbe.x - 0.36f +
                                        0.22f *
                                            (detailProbe.z - 0.50f) +
                                        0.25f * warpA +
                                        0.10f * detailProbe.y;
                                    const float sheetB =
                                        detailProbe.x - 0.50f -
                                        0.18f *
                                            (detailProbe.z - 0.50f) +
                                        0.27f * warpB -
                                        0.09f * detailProbe.y;
                                    const float sheetC =
                                        detailProbe.x - 0.64f +
                                        0.12f *
                                            (detailProbe.z - 0.50f) +
                                        0.24f * warpC +
                                        0.13f * detailProbe.y;
                                    const float width =
                                        0.012f +
                                        0.014f *
                                            ribbonWidthSignal;
                                    detailRidge =
                                        std::max(
                                            std::exp(
                                                -square(sheetA / width)),
                                            std::max(
                                                0.92f *
                                                    std::exp(
                                                        -square(
                                                            sheetB /
                                                            (0.90f *
                                                             width))),
                                                0.82f *
                                                    std::exp(
                                                        -square(
                                                            sheetC /
                                                            (0.84f *
                                                             width)))));
                                }
                                detailRidge *= ribbonOpening;
                                const float detailedBody =
                                    reconstructedSupport * detailFalloff *
                                    (0.18f +
                                     0.82f *
                                         smoothstep(
                                             0.16f, 0.84f,
                                             advectedDetail));
                                const float detailedRibbon =
                                    reconstructedSupport * detailFalloff *
                                    detailRidge;
                                const float detailedReaction =
                                    wildfire
                                        ? 0.0f
                                        : 0.95f * detailedRibbon;
                                const float foldedSheetReaction =
                                    wildfire
                                        ? 0.075f *
                                                  std::pow(
                                                      wildfireSheetPatch,
                                                      0.78f) +
                                              0.045f *
                                                  std::pow(
                                                      wildfireSheetPatch,
                                                      1.70f)
                                        : 0.0f;
                                const float wildfireCoreReaction =
                                    wildfire
                                        ? 0.100f *
                                                  std::pow(
                                                      wildfireCorePatch,
                                                      0.86f) +
                                              0.060f *
                                                  std::pow(
                                                      wildfireCorePatch,
                                                      2.00f)
                                        : 0.0f;
                                const float wildfireBodyReaction =
                                    wildfire
                                        ? 0.006f *
                                              std::pow(
                                                  wildfireBodyPatch,
                                                  0.90f)
                                        : 0.0f;
                                wildfireSheetHeatStrength =
                                    wildfire ? wildfireSheetPatch : 0.0f;
                                wildfireCoreHeatStrength =
                                    wildfire ? wildfireCorePatch : 0.0f;
                                wildfireBodyHeatStrength =
                                    wildfire ? wildfireBodyPatch : 0.0f;
                                const float attachedBody =
                                    reconstructedSupport * bodyFalloff *
                                    rootGate *
                                    ((wildfire ? 0.32f : 0.16f) +
                                     (wildfire ? 0.68f : 0.84f) *
                                         std::pow(broadMixSheet, 1.18f));
                                if (wildfire) {
                                    // Flames and incandescent soot are
                                    // separate fields. Keep the soot carrier
                                    // in a short fuel-attached layer; tall
                                    // structure comes from the folded reaction
                                    // sheets above it. Advected canopy soot is
                                    // outside this gate and remains cool.
                                    const float carrierHeightSignal =
                                        0.5f +
                                        0.5f *
                                            fractalNoise(
                                                glm::vec3(
                                                    rootProbe.x *
                                                        6.7f,
                                                    4.6f,
                                                    rootProbe.z *
                                                        7.9f),
                                                settings.seed + 3917);
                                    const float carrierHeight =
                                        0.018f +
                                        0.030f *
                                            (0.30f +
                                             0.70f *
                                                 carrierHeightSignal);
                                    const float carrierHeightRatio =
                                        heightAboveRoot /
                                        std::max(
                                            carrierHeight,
                                            1.0e-4f);
                                    const float carrierY =
                                        rootGate *
                                        std::exp(
                                            -std::pow(
                                                carrierHeightRatio,
                                                1.80f)) *
                                        (1.0f -
                                         smoothstep(
                                             0.78f, 1.05f,
                                             carrierHeightRatio));
                                    const float sootCarrierEnvelope =
                                        saturate(
                                            rootIgnition * carrierY *
                                            (0.35f +
                                             0.65f *
                                                 saturate(
                                                     rootSource.solidFuel)) *
                                            (0.62f +
                                             0.38f *
                                                 wildfireDetailSupport));
                                    const float sootCarrierGate =
                                        smoothstep(
                                            0.035f, 0.30f,
                                            sootCarrierEnvelope);
                                    wildfireSootCarrierHeat =
                                        sootCarrierGate *
                                        (900.0f +
                                         600.0f *
                                             std::sqrt(
                                                 sootCarrierEnvelope));
                                    // Oxygen-rich active combustion burns
                                    // down soot locally without weakening the
                                    // cool, transported smoke canopy.
                                    soot *=
                                        1.0f -
                                        0.40f *
                                            sootCarrierGate;
                                    soot =
                                        std::max(
                                            soot,
                                            0.15f *
                                                sootCarrierEnvelope);
                                }
                                const float internalRibbon =
                                    reconstructedSupport *
                                    boundedRibbonFalloff *
                                    std::max(
                                        std::pow(zeroCrossing, 0.82f),
                                        0.72f *
                                            std::pow(ribbon, 1.32f)) *
                                    (wildfire ? 1.0f :
                                                (0.35f +
                                                 0.65f *
                                                     ribbonOpening));
                                const float connectedRoot =
                                    wildfire
                                        ? reconstructedSupport * 0.018f *
                                              rootGate *
                                              std::exp(
                                                  -square(
                                                      heightAboveRoot /
                                                      0.016f)) *
                                              (0.75f +
                                               0.25f *
                                                   smoothstep(
                                                       0.20f, 0.78f,
                                                       detailSegmentation))
                                        : 0.22f *
                                              woodRootUnionCoverage *
                                              rootUnionFalloff *
                                              rootGate *
                                              (0.12f +
                                               0.88f *
                                                   smoothstep(
                                                       0.40f, 0.72f,
                                                       rootUnionNoise));
                                reaction =
                                    std::max(
                                        reaction,
                                        std::max(
                                            connectedRoot,
                                            std::max(
                                                std::max(
                                                    (wildfire ? 0.0f :
                                                                0.040f) *
                                                        attachedBody,
                                                    (wildfire ? 0.0f :
                                                                0.060f) *
                                                        detailedBody),
                                                std::max(
                                                    (wildfire ? 0.0f :
                                                                0.30f) *
                                                        internalRibbon,
                                                    std::max(
                                                        detailedReaction,
                                                        std::max(
                                                            foldedSheetReaction,
                                                            std::max(
                                                                wildfireCoreReaction,
                                                                wildfireBodyReaction)))))));
                            }
                            reaction = saturate(
                                reaction * settings.reactionScale);
                            const bool wildfirePreset =
                                settings.preset ==
                                CombustionPreset::Wildfire;
                            float effectiveHeat =
                                std::max(0.0f, cell.heat);
                            if (wildfirePreset) {
                                // The coarse fallback solve needs its heat for
                                // buoyancy, but its render-time residual forms
                                // smooth incandescent lobes. Let the short soot
                                // carrier and folded reaction sheets determine
                                // visible fire temperature instead; transported
                                // smoke remains cool.
                                effectiveHeat =
                                    std::min(effectiveHeat, 650.0f);
                                effectiveHeat =
                                    std::max(effectiveHeat,
                                             wildfireSootCarrierHeat);
                                const float bodyStrength =
                                    saturate(
                                        wildfireBodyHeatStrength);
                                const float bodyHeat =
                                    1050.0f *
                                    std::pow(
                                        bodyStrength,
                                        0.42f);
                                effectiveHeat =
                                    std::max(effectiveHeat,
                                             bodyHeat);
                                const float sheetStrength =
                                    saturate(
                                        wildfireSheetHeatStrength);
                                const float sheetHeat =
                                    2250.0f *
                                    std::pow(
                                        sheetStrength,
                                        0.40f);
                                effectiveHeat =
                                    std::max(effectiveHeat,
                                             sheetHeat);
                                const float coreStrength =
                                    saturate(
                                        wildfireCoreHeatStrength);
                                const float coreHeat =
                                    2920.0f *
                                    std::pow(
                                        coreStrength,
                                        0.32f);
                                effectiveHeat =
                                    std::max(effectiveHeat,
                                             coreHeat);
                            } else if (settings.preset !=
                                       CombustionPreset::Candle) {
                                // Active combustion and temperature must be
                                // thermodynamically registered. The flow
                                // solve supplies the broad heat carrier; this
                                // lower bound keeps newly reconstructed
                                // stoichiometric sheets hot while leaving
                                // advected smoke cool and non-emissive.
                                if (reaction > 1e-4f) {
                                    const float sheetHeat =
                                        1050.0f +
                                        1180.0f *
                                            std::sqrt(reaction);
                                    effectiveHeat =
                                        std::max(effectiveHeat,
                                                 sheetHeat);
                                }
                            }
                            const float temperature =
                                kAmbientTemperature +
                                effectiveHeat *
                                    settings.temperatureScale;

                            const int localIndex = brickSampleIndex(
                                sampleX, sampleY, sampleZ);
                            combustion[localIndex] =
                                glm::vec4(density, soot, fuel, reaction);
                            thermalFlow[localIndex] =
                                glm::vec4(temperature, cell.velocity.x,
                                          cell.velocity.y,
                                          cell.velocity.z);

                            meta.maxDensity =
                                std::max(meta.maxDensity, density);
                            meta.maxSoot =
                                std::max(meta.maxSoot, soot);
                            meta.maxFuel =
                                std::max(meta.maxFuel, fuel);
                            meta.maxReaction =
                                std::max(meta.maxReaction, reaction);
                            meta.maxTemperature =
                                std::max(meta.maxTemperature,
                                         temperature);
                            meta.maxFieldExtinction =
                                std::max(meta.maxFieldExtinction,
                                         density + soot);
                            const float normalizedHeat =
                                std::max(0.0f,
                                         (temperature -
                                          kAmbientTemperature) /
                                             2100.0f);
                            activity = std::max(
                                activity,
                                std::max(
                                    std::max(density, soot),
                                    std::max(
                                        std::max(fuel, reaction),
                                        normalizedHeat)));
                        }
                    }
                }

                if (activity <= settings.activeThreshold) {
                    continue;
                }

                if (grid.bricks.size() >=
                        static_cast<std::size_t>(
                            std::numeric_limits<int>::max()) ||
                    grid.combustionSamples.size() >
                        static_cast<std::size_t>(
                            std::numeric_limits<std::uint32_t>::max() -
                            kVolumeBrickSampleCount) ||
                    grid.cellEmissionCdf.size() >
                        static_cast<std::size_t>(
                            std::numeric_limits<std::uint32_t>::max() -
                            kVolumeBrickCellCount)) {
                    throw std::overflow_error(
                        "sparse volume grid exceeds 32-bit device offsets");
                }

                const int activeIndex =
                    static_cast<int>(grid.bricks.size());
                meta.sampleOffset = static_cast<std::uint32_t>(
                    grid.combustionSamples.size());
                meta.cellCdfOffset = static_cast<std::uint32_t>(
                    grid.cellEmissionCdf.size());

                std::array<float, kVolumeBrickCellCount> cellPower{};
                float brickPower = 0.0f;
                for (int cellZ = 0; cellZ < kVolumeBrickSize; ++cellZ) {
                    for (int cellY = 0; cellY < kVolumeBrickSize;
                         ++cellY) {
                        for (int cellX = 0; cellX < kVolumeBrickSize;
                             ++cellX) {
                            const int cellIndex =
                                brickCellIndex(cellX, cellY, cellZ);
                            const float power =
                                std::max(
                                    0.0f,
                                    emissionProxy(combustion, thermalFlow,
                                                  cellX, cellY, cellZ)) *
                                grid.cellVolume;
                            cellPower[cellIndex] = power;
                            brickPower += power;
                        }
                    }
                }
                meta.emissionPower = brickPower;

                float cumulative = 0.0f;
                if (brickPower > 0.0f) {
                    for (float power : cellPower) {
                        cumulative += power / brickPower;
                        grid.cellEmissionCdf.push_back(
                            std::min(cumulative, 1.0f));
                    }
                } else {
                    for (int i = 0; i < kVolumeBrickCellCount; ++i) {
                        grid.cellEmissionCdf.push_back(
                            static_cast<float>(i + 1) /
                            static_cast<float>(
                                kVolumeBrickCellCount));
                    }
                }
                grid.cellEmissionCdf.back() = 1.0f;

                grid.pageTable[pageIndex(grid.brickResolution, brickX,
                                         brickY, brickZ)] = activeIndex;
                grid.bricks.push_back(meta);
                grid.combustionSamples.insert(
                    grid.combustionSamples.end(), combustion.begin(),
                    combustion.end());
                grid.thermalFlowSamples.insert(
                    grid.thermalFlowSamples.end(), thermalFlow.begin(),
                    thermalFlow.end());
            }
        }
    }

    dilateBrickMaxima26(grid);

    for (const VolumeBrickMeta &brick : grid.bricks) {
        grid.totalEmissionPower += brick.emissionPower;
    }
    float cumulative = 0.0f;
    if (grid.totalEmissionPower > 0.0f) {
        for (const VolumeBrickMeta &brick : grid.bricks) {
            cumulative += brick.emissionPower /
                          grid.totalEmissionPower;
            grid.brickEmissionCdf.push_back(
                std::min(cumulative, 1.0f));
        }
    } else if (!grid.bricks.empty()) {
        for (std::size_t i = 0; i < grid.bricks.size(); ++i) {
            grid.brickEmissionCdf.push_back(
                static_cast<float>(i + 1) /
                static_cast<float>(grid.bricks.size()));
        }
    }
    if (!grid.brickEmissionCdf.empty()) {
        grid.brickEmissionCdf.back() = 1.0f;
    }

    std::string validationError;
    if (!validateSparseCombustionGrid(grid, &validationError)) {
        throw std::runtime_error("generated sparse volume is invalid: " +
                                 validationError);
    }
    return grid;
}

bool validateSparseCombustionGrid(const HostSparseVolumeGrid &grid,
                                  std::string *error) {
    if (grid.brickSize != kVolumeBrickSize) {
        return failValidation(error, "brickSize is not 8");
    }
    if (grid.cellResolution.x <= 0 || grid.cellResolution.y <= 0 ||
        grid.cellResolution.z <= 0 ||
        grid.cellResolution.x % kVolumeBrickSize != 0 ||
        grid.cellResolution.y % kVolumeBrickSize != 0 ||
        grid.cellResolution.z % kVolumeBrickSize != 0) {
        return failValidation(error,
                              "cell resolution is invalid or unaligned");
    }
    if (grid.brickResolution !=
        grid.cellResolution / kVolumeBrickSize) {
        return failValidation(error, "brick resolution is inconsistent");
    }
    if (!finiteVec3(grid.localBoundsMin) ||
        !finiteVec3(grid.localBoundsMax) ||
        glm::any(glm::lessThanEqual(grid.localBoundsMax,
                                    grid.localBoundsMin))) {
        return failValidation(error, "local bounds are invalid");
    }
    if (!std::isfinite(grid.cellVolume) || grid.cellVolume <= 0.0f ||
        !std::isfinite(grid.totalEmissionPower) ||
        grid.totalEmissionPower < 0.0f) {
        return failValidation(error, "grid scalar metadata is invalid");
    }

    const std::size_t expectedPageCount =
        static_cast<std::size_t>(grid.brickResolution.x) *
        static_cast<std::size_t>(grid.brickResolution.y) *
        static_cast<std::size_t>(grid.brickResolution.z);
    if (grid.pageTable.size() != expectedPageCount) {
        return failValidation(error, "page-table size is inconsistent");
    }
    if (grid.combustionSamples.size() !=
            grid.bricks.size() * kVolumeBrickSampleCount ||
        grid.thermalFlowSamples.size() !=
            grid.bricks.size() * kVolumeBrickSampleCount ||
        grid.cellEmissionCdf.size() !=
            grid.bricks.size() * kVolumeBrickCellCount ||
        grid.brickEmissionCdf.size() != grid.bricks.size()) {
        return failValidation(error,
                              "sparse payload size is inconsistent");
    }

    std::vector<int> pageReferences(grid.bricks.size(), 0);
    for (int page : grid.pageTable) {
        if (page < -1 ||
            page >= static_cast<int>(grid.bricks.size())) {
            return failValidation(error,
                                  "page-table entry is out of range");
        }
        if (page >= 0) {
            ++pageReferences[static_cast<std::size_t>(page)];
        }
    }

    float recomputedTotalPower = 0.0f;
    std::vector<VolumeBrickMeta> payloadMaxima(grid.bricks.size());
    for (std::size_t brickIndex = 0; brickIndex < grid.bricks.size();
         ++brickIndex) {
        const VolumeBrickMeta &meta = grid.bricks[brickIndex];
        if (pageReferences[brickIndex] != 1) {
            return failValidation(
                error, "active brick is not referenced exactly once");
        }
        if (meta.brickX < 0 ||
            meta.brickX >= grid.brickResolution.x ||
            meta.brickY < 0 ||
            meta.brickY >= grid.brickResolution.y ||
            meta.brickZ < 0 ||
            meta.brickZ >= grid.brickResolution.z ||
            grid.pageTable[pageIndex(grid.brickResolution, meta.brickX,
                                     meta.brickY, meta.brickZ)] !=
                static_cast<int>(brickIndex)) {
            return failValidation(error,
                                  "brick coordinate/page mapping is invalid");
        }
        if (meta.sampleOffset !=
                brickIndex * kVolumeBrickSampleCount ||
            meta.cellCdfOffset !=
                brickIndex * kVolumeBrickCellCount) {
            return failValidation(error,
                                  "brick payload offset is invalid");
        }
        if (!std::isfinite(meta.maxDensity) ||
            !std::isfinite(meta.maxSoot) ||
            !std::isfinite(meta.maxFuel) ||
            !std::isfinite(meta.maxReaction) ||
            !std::isfinite(meta.maxTemperature) ||
            !std::isfinite(meta.maxFieldExtinction) ||
            !std::isfinite(meta.emissionPower) ||
            meta.maxDensity < 0.0f || meta.maxSoot < 0.0f ||
            meta.maxFuel < 0.0f || meta.maxReaction < 0.0f ||
            meta.maxTemperature < 0.0f ||
            meta.maxFieldExtinction < 0.0f ||
            meta.emissionPower < 0.0f) {
            return failValidation(error,
                                  "brick metadata is non-finite or negative");
        }

        float maxDensity = 0.0f;
        float maxSoot = 0.0f;
        float maxFuel = 0.0f;
        float maxReaction = 0.0f;
        float maxTemperature = 0.0f;
        float maxFieldExtinction = 0.0f;
        for (int sample = 0; sample < kVolumeBrickSampleCount;
             ++sample) {
            const std::size_t index =
                static_cast<std::size_t>(meta.sampleOffset) + sample;
            const glm::vec4 combustion =
                grid.combustionSamples[index];
            const glm::vec4 thermal = grid.thermalFlowSamples[index];
            if (!finiteVec4(combustion) || !finiteVec4(thermal) ||
                combustion.x < 0.0f || combustion.y < 0.0f ||
                combustion.z < 0.0f || combustion.w < 0.0f ||
                thermal.x < 0.0f) {
                return failValidation(
                    error, "sample payload is non-finite or negative");
            }
            maxDensity = std::max(maxDensity, combustion.x);
            maxSoot = std::max(maxSoot, combustion.y);
            maxFuel = std::max(maxFuel, combustion.z);
            maxReaction = std::max(maxReaction, combustion.w);
            maxTemperature = std::max(maxTemperature, thermal.x);
            maxFieldExtinction =
                std::max(maxFieldExtinction,
                         combustion.x + combustion.y);
        }
        const float tolerance = 2.0e-6f;
        if (meta.maxDensity + tolerance < maxDensity ||
            meta.maxSoot + tolerance < maxSoot ||
            meta.maxFuel + tolerance < maxFuel ||
            meta.maxReaction + tolerance < maxReaction ||
            meta.maxTemperature + tolerance < maxTemperature ||
            meta.maxFieldExtinction + tolerance <
                maxFieldExtinction) {
            return failValidation(
                error, "brick maxima are not conservative over the halo");
        }
        VolumeBrickMeta &raw = payloadMaxima[brickIndex];
        raw.maxDensity = maxDensity;
        raw.maxSoot = maxSoot;
        raw.maxFuel = maxFuel;
        raw.maxReaction = maxReaction;
        raw.maxTemperature = maxTemperature;
        raw.maxFieldExtinction = maxFieldExtinction;

        const float *cellCdf =
            grid.cellEmissionCdf.data() + meta.cellCdfOffset;
        std::ostringstream cdfName;
        cdfName << "cell emission CDF for brick " << brickIndex;
        if (!validateNormalizedCdf(cellCdf, kVolumeBrickCellCount,
                                   cdfName.str(), error)) {
            return false;
        }
        const float brickLower =
            brickIndex > 0 ? grid.brickEmissionCdf[brickIndex - 1]
                           : 0.0f;
        const float brickProbability =
            grid.brickEmissionCdf[brickIndex] - brickLower;
        float cellLower = 0.0f;
        for (int cellZ = 0; cellZ < kVolumeBrickSize; ++cellZ) {
            for (int cellY = 0; cellY < kVolumeBrickSize; ++cellY) {
                for (int cellX = 0; cellX < kVolumeBrickSize; ++cellX) {
                    float maxReaction = 0.0f;
                    float maxSoot = 0.0f;
                    float maxTemperature = 0.0f;
                    for (int dz = 0; dz <= 1; ++dz) {
                        for (int dy = 0; dy <= 1; ++dy) {
                            for (int dx = 0; dx <= 1; ++dx) {
                                const std::size_t sample =
                                    static_cast<std::size_t>(
                                        meta.sampleOffset) +
                                    static_cast<std::size_t>(
                                        brickSampleIndex(
                                            cellX + dx, cellY + dy,
                                            cellZ + dz));
                                maxReaction =
                                    std::max(maxReaction,
                                             grid.combustionSamples[sample]
                                                 .w);
                                maxSoot =
                                    std::max(maxSoot,
                                             grid.combustionSamples[sample]
                                                 .y);
                                maxTemperature =
                                    std::max(maxTemperature,
                                             grid.thermalFlowSamples[sample]
                                                 .x);
                            }
                        }
                    }
                    const bool hasEmissionSupport =
                        maxReaction > 0.0f ||
                        (maxSoot > 0.0f &&
                         maxTemperature >
                             kVolumeSootEmissionOnsetTemperature);
                    const int cellIndex =
                        brickCellIndex(cellX, cellY, cellZ);
                    const float cellProbability =
                        cellCdf[cellIndex] - cellLower;
                    cellLower = cellCdf[cellIndex];
                    if (hasEmissionSupport &&
                        (brickProbability <= 0.0f ||
                         cellProbability <= 0.0f)) {
                        std::ostringstream stream;
                        stream << "emissive cell has zero CDF support in "
                                  "brick "
                               << brickIndex << " at cell " << cellIndex;
                        return failValidation(error, stream.str());
                    }
                }
            }
        }
        recomputedTotalPower += meta.emissionPower;
    }

    // Independently verify the metadata dilation against maxima recomputed
    // from each neighbor's actual 9^3 payload. This prevents a stale or
    // traversal-order-dependent dilation pass from silently weakening local
    // null-collision majorants near brick boundaries.
    const float neighborTolerance = 2.0e-6f;
    for (std::size_t brickIndex = 0; brickIndex < grid.bricks.size();
         ++brickIndex) {
        const VolumeBrickMeta &meta = grid.bricks[brickIndex];
        for (int dz = -1; dz <= 1; ++dz) {
            const int z = meta.brickZ + dz;
            if (z < 0 || z >= grid.brickResolution.z) {
                continue;
            }
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = meta.brickY + dy;
                if (y < 0 || y >= grid.brickResolution.y) {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = meta.brickX + dx;
                    if (x < 0 || x >= grid.brickResolution.x) {
                        continue;
                    }
                    const int neighborIndex =
                        grid.pageTable[pageIndex(grid.brickResolution, x,
                                                 y, z)];
                    if (neighborIndex < 0) {
                        continue;
                    }
                    const VolumeBrickMeta &neighbor =
                        payloadMaxima[static_cast<std::size_t>(
                            neighborIndex)];
                    if (meta.maxDensity + neighborTolerance <
                            neighbor.maxDensity ||
                        meta.maxSoot + neighborTolerance <
                            neighbor.maxSoot ||
                        meta.maxFuel + neighborTolerance <
                            neighbor.maxFuel ||
                        meta.maxReaction + neighborTolerance <
                            neighbor.maxReaction ||
                        meta.maxTemperature + neighborTolerance <
                            neighbor.maxTemperature ||
                        meta.maxFieldExtinction + neighborTolerance <
                            neighbor.maxFieldExtinction) {
                        return failValidation(
                            error,
                            "brick maxima are not conservative over the "
                            "26-neighbor dilation");
                    }
                }
            }
        }
    }

    if (!grid.bricks.empty() &&
        !validateNormalizedCdf(grid.brickEmissionCdf.data(),
                               grid.brickEmissionCdf.size(),
                               "brick emission CDF", error)) {
        return false;
    }
    const float powerTolerance =
        1.0e-5f * std::max(1.0f, grid.totalEmissionPower);
    if (std::abs(recomputedTotalPower - grid.totalEmissionPower) >
        powerTolerance) {
        return failValidation(error,
                              "total emission power is inconsistent");
    }
    return true;
}
