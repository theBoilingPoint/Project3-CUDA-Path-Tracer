#include "scene.h"

#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "spectrumData.h" // illuminantRGB / blackbodyLuminanceNorm

#define USE_SELF_LOADED_TEXTURES 1

static int parseVolumeDebugMode(const std::string &name) {
    if (name == "None")
        return VOLUME_DEBUG_NONE;
    if (name == "Temperature")
        return VOLUME_DEBUG_TEMPERATURE;
    if (name == "Density")
        return VOLUME_DEBUG_DENSITY;
    if (name == "Fuel")
        return VOLUME_DEBUG_FUEL;
    if (name == "Soot")
        return VOLUME_DEBUG_SOOT;
    if (name == "Reaction")
        return VOLUME_DEBUG_REACTION;
    if (name == "Emission")
        return VOLUME_DEBUG_EMISSION;
    if (name == "SigmaA")
        return VOLUME_DEBUG_SIGMA_A;
    if (name == "SigmaS")
        return VOLUME_DEBUG_SIGMA_S;
    if (name == "SigmaT")
        return VOLUME_DEBUG_SIGMA_T;
    if (name == "Velocity")
        return VOLUME_DEBUG_VELOCITY;
    if (name == "Majorant")
        return VOLUME_DEBUG_MAJORANT;
    if (name == "NullRate")
        return VOLUME_DEBUG_NULL_RATE;
    if (name == "EventCount")
        return VOLUME_DEBUG_EVENT_COUNT;
    if (name == "DirectVolume")
        return VOLUME_DEBUG_DIRECT_VOLUME;
    if (name == "IndirectVolume")
        return VOLUME_DEBUG_INDIRECT_VOLUME;
    if (name == "SurfaceFire")
        return VOLUME_DEBUG_SURFACE_FIRE;
    std::cerr << "Unknown volume DEBUG mode \"" << name << "\"." << std::endl;
    exit(-1);
}

static CombustionPreset parseCombustionPreset(const std::string &name) {
    if (name == "Candle")
        return CombustionPreset::Candle;
    if (name == "WoodFire")
        return CombustionPreset::WoodFire;
    if (name == "Wildfire")
        return CombustionPreset::Wildfire;
    std::cerr << "Unknown VOLUME_GRID PRESET \"" << name
              << "\" (expected Candle, WoodFire, or Wildfire)." << std::endl;
    exit(-1);
}

static void printVolumeFieldSummary(const HostSparseVolumeGrid &grid) {
    float maxDensity = 0.0f;
    float maxSoot = 0.0f;
    float maxReaction = 0.0f;
    float maxTemperature = 0.0f;
    glm::vec3 reactionMin(1.0f);
    glm::vec3 reactionMax(0.0f);
    bool hasReaction = false;
    for (const VolumeBrickMeta &brick : grid.bricks) {
        maxDensity = std::max(maxDensity, brick.maxDensity);
        maxSoot = std::max(maxSoot, brick.maxSoot);
        maxReaction = std::max(maxReaction, brick.maxReaction);
        maxTemperature = std::max(maxTemperature, brick.maxTemperature);
        for (int z = 0; z < kVolumeBrickSampleSize; ++z) {
            for (int y = 0; y < kVolumeBrickSampleSize; ++y) {
                for (int x = 0; x < kVolumeBrickSampleSize; ++x) {
                    const int local =
                        x + kVolumeBrickSampleSize *
                                (y + kVolumeBrickSampleSize * z);
                    const glm::vec4 combustion =
                        grid.combustionSamples[brick.sampleOffset + local];
                    if (combustion.w <= 0.01f) {
                        continue;
                    }
                    const glm::vec3 q =
                        glm::vec3(brick.brickX * grid.brickSize + x,
                                  brick.brickY * grid.brickSize + y,
                                  brick.brickZ * grid.brickSize + z) /
                        glm::vec3(grid.cellResolution);
                    reactionMin = glm::min(reactionMin, q);
                    reactionMax = glm::max(reactionMax, q);
                    hasReaction = true;
                }
            }
        }
    }
    printf("Fields: max density %.3f, soot %.3f, reaction %.3f, "
           "temperature %.0f K",
           maxDensity, maxSoot, maxReaction, maxTemperature);
    if (hasReaction) {
        printf(", reaction q-bounds [%.2f %.2f %.2f]-[%.2f %.2f %.2f]",
               reactionMin.x, reactionMin.y, reactionMin.z, reactionMax.x,
               reactionMax.y, reactionMax.z);
    }
    printf("\n");
}

static float luminance(const glm::vec3 &rgb) {
    return glm::dot(rgb, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

// Compute once on the host instead of re-summing mesh triangles in every NEE
// and hit-light MIS evaluation. The affine-sphere value is only an importance
// proxy; its conditional sample PDF is evaluated with the exact area
// Jacobian on the GPU.
static float geomSurfaceArea(const Geom &geom, const MeshData &mesh) {
    const glm::vec3 ex = glm::vec3(geom.transform.transform[0]);
    const glm::vec3 ey = glm::vec3(geom.transform.transform[1]);
    const glm::vec3 ez = glm::vec3(geom.transform.transform[2]);
    if (geom.type == CUBE) {
        return 2.0f * (glm::length(glm::cross(ey, ez)) +
                       glm::length(glm::cross(ex, ez)) +
                       glm::length(glm::cross(ex, ey)));
    }
    if (geom.type == SPHERE) {
        const float a = 0.5f * glm::length(ex);
        const float b = 0.5f * glm::length(ey);
        const float c = 0.5f * glm::length(ez);
        constexpr float p = 1.6075f;
        const float mean =
            (std::pow(a * b, p) + std::pow(a * c, p) +
             std::pow(b * c, p)) /
            3.0f;
        return 4.0f * glm::pi<float>() * std::pow(mean, 1.0f / p);
    }
    if (geom.type == MESH && mesh.triangles != nullptr) {
        float area = 0.0f;
        for (int i = 0; i < mesh.numTriangles; ++i) {
            const Triangle &triangle = mesh.triangles[i];
            const glm::vec3 p0 = glm::vec3(
                geom.transform.transform * glm::vec4(triangle.points[0], 1));
            const glm::vec3 p1 = glm::vec3(
                geom.transform.transform * glm::vec4(triangle.points[1], 1));
            const glm::vec3 p2 = glm::vec3(
                geom.transform.transform * glm::vec4(triangle.points[2], 1));
            area += 0.5f * glm::length(glm::cross(p1 - p0, p2 - p0));
        }
        return area;
    }
    return 0.0f;
}

// Parse an optional "SPECTRUM" field on an emitter or delta light:
//   "SPECTRUM": "D65" | "A" | "E"          (named illuminants)
//   "SPECTRUM": {"BLACKBODY": <kelvin>}    (Planck radiator)
// In SPECTRAL builds the light emits that SPD (times its RGB tint, times
// EMITTANCE/INTENSITY); blackbodyNorm is filled so the device can normalize
// Planck to unit luminance. In RGB builds the illuminant's RGB equivalent is
// folded into `rgb` instead, approximating the spectral look.
template <typename JsonT>
static void parseSpectrum(const JsonT &p, glm::vec3 &rgb, int &spectrumType,
                          float &blackbodyTemp, float &blackbodyNorm) {
    spectrumType = SPECTRUM_NONE;
    blackbodyTemp = 0.0f;
    blackbodyNorm = 0.0f;
    if (!p.contains("SPECTRUM")) {
        return;
    }
    const auto &s = p["SPECTRUM"];
    if (s.is_string()) {
        const std::string name = s;
        if (name == "D65") {
            spectrumType = SPECTRUM_D65;
        } else if (name == "A") {
            spectrumType = SPECTRUM_A;
        } else if (name == "E") {
            spectrumType = SPECTRUM_E;
        } else {
            printf("Unknown SPECTRUM \"%s\" (expected \"D65\", \"A\", \"E\" or "
                   "{\"BLACKBODY\": kelvin}).\n",
                   name.c_str());
            exit(-1);
        }
    } else if (s.is_object() && s.contains("BLACKBODY")) {
        float kelvin = (float)s["BLACKBODY"];
        if (kelvin <= 0.0f) {
            printf("SPECTRUM BLACKBODY temperature must be > 0 K.\n");
            exit(-1);
        }
        spectrumType = SPECTRUM_BLACKBODY;
        blackbodyTemp = kelvin;
        blackbodyNorm = blackbodyLuminanceNorm(kelvin);
    } else {
        printf("Malformed SPECTRUM field (expected \"D65\", \"A\", \"E\" or "
               "{\"BLACKBODY\": kelvin}).\n");
        exit(-1);
    }
#if !SPECTRAL
    // RGB build: approximate the illuminant by tinting the light's RGB color.
    rgb *= illuminantRGB(spectrumType, blackbodyTemp);
    spectrumType = SPECTRUM_NONE;
    blackbodyTemp = 0.0f;
    blackbodyNorm = 0.0f;
#endif
}

Mesh::Mesh() {}

Mesh::~Mesh() {
    faces.clear();
    verts.clear();
    normals.clear();
    indices.clear();
    uvs.clear();
    albedoTextures.clear();
    normalTextures.clear();
    bumpTextures.clear();
}

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json") {
        loadFromJSON(filename);
        return;
    } else {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

Scene::~Scene() {
    // Clean up texture data (if not already cleaned up by pathtraceFree)
    for (auto &tex : albedoTextures) {
        glm::vec4 *ptr = std::get<0>(tex);
        // Check for valid pointer (not nullptr and not obviously corrupted)
        if (ptr != nullptr &&
            ptr != reinterpret_cast<glm::vec4 *>(0xFFFFFFFFFFFFFFFFULL)) {
            delete[] ptr;
        }
    }
    for (auto &tex : normalTextures) {
        glm::vec4 *ptr = std::get<0>(tex);
        if (ptr != nullptr &&
            ptr != reinterpret_cast<glm::vec4 *>(0xFFFFFFFFFFFFFFFFULL)) {
            delete[] ptr;
        }
    }
    for (auto &tex : bumpTextures) {
        glm::vec4 *ptr = std::get<0>(tex);
        if (ptr != nullptr &&
            ptr != reinterpret_cast<glm::vec4 *>(0xFFFFFFFFFFFFFFFFULL)) {
            delete[] ptr;
        }
    }

    // Note: the host mesh arrays should already be cleaned up by pathtraceFree;
    // only free them here if pathtraceFree wasn't called. `geomMeshData` is the
    // sole owner (`lightMeshData` holds non-owning copies), so we free only it.
    // delete[] on nullptr is a no-op, so double cleanup is safe.
    for (MeshData &md : geomMeshData) {
        delete[] md.triangles;
        md.triangles = nullptr;
        delete[] md.nodes;
        md.nodes = nullptr;
    }

    // Free the host-side environment map pixels (uploaded to a CUDA texture
    // during deviceSceneInit; delete[] on nullptr is a no-op).
    delete[] envMap;
    envMap = nullptr;
}

// Builds a PBRT-style piecewise-constant 2D distribution over the loaded
// equirectangular env map for importance sampling (NEE + MIS). The per-pixel
// importance is luminance * sin(theta), where theta = pi * (row + 0.5) / height
// is the polar angle; the sin(theta) factor is the lat-long solid-angle
// Jacobian, so bright pixels crammed near the poles are not over-weighted.
//
// Layout matches Distribution2D: one conditional CDF per row over columns
// (u | v), and one marginal CDF over rows (v). Both are normalized to [0, 1].
// With that normalization, all sampling/pdf math on the device reduces to
// scaled CDF differences (see envPdf/sampleEnvDirection in pathtrace.cu), so we
// don't need to also upload the raw function or its integrals.
void Scene::buildEnvDistribution() {
    const int w = envMapSize.x;
    const int h = envMapSize.y;
    if (!hasEnvMap || envMap == nullptr || w <= 0 || h <= 0) {
        return;
    }

    envConditionalCdf.assign((size_t)h * (w + 1), 0.0f);
    envMarginalCdf.assign((size_t)h + 1, 0.0f);

    // Per-row (marginal) function values = each row's unnormalized integral.
    std::vector<float> marginalFunc(h, 0.0f);

    for (int y = 0; y < h; ++y) {
        const float theta = PI * ((float)y + 0.5f) / (float)h;
        const float sinTheta = sinf(theta);
        float *cdf = &envConditionalCdf[(size_t)y * (w + 1)];

        cdf[0] = 0.0f;
        for (int x = 0; x < w; ++x) {
            const glm::vec4 &px = envMap[(size_t)y * w + x];
            const float lum = 0.2126f * px.r + 0.7152f * px.g + 0.0722f * px.b;
            const float f = lum * sinTheta;     // importance
            cdf[x + 1] = cdf[x] + f / (float)w; // running integral / n
        }

        const float rowInt = cdf[w];
        marginalFunc[y] = rowInt;
        if (rowInt > 0.0f) {
            for (int x = 1; x <= w; ++x) {
                cdf[x] /= rowInt; // normalize row CDF to [0, 1]
            }
        } else {
            // Black row: fall back to a uniform CDF so sampling stays valid.
            for (int x = 1; x <= w; ++x) {
                cdf[x] = (float)x / (float)w;
            }
        }
    }

    envMarginalCdf[0] = 0.0f;
    for (int y = 0; y < h; ++y) {
        envMarginalCdf[y + 1] = envMarginalCdf[y] + marginalFunc[y] / (float)h;
    }
    const float total = envMarginalCdf[h];
    if (total > 0.0f) {
        for (int y = 1; y <= h; ++y) {
            envMarginalCdf[y] /= total;
        }
    } else {
        // Fully black map: uniform marginal (importance sampling is a no-op).
        for (int y = 1; y <= h; ++y) {
            envMarginalCdf[y] = (float)y / (float)h;
        }
    }
}

void Scene::loadMesh(const std::string &filepath, Mesh &mesh) {
    if (endsWith(filepath, ".obj")) {
        printf("Loading OBJ file: %s\n", filepath.c_str());
        loadOBJ(filepath, mesh.faces, mesh.verts, mesh.normals, mesh.uvs,
                mesh.indices);
    } else if (endsWith(filepath, ".gltf") || endsWith(filepath, ".glb")) {
        loadGLTFOrGLB(filepath, mesh.faces, mesh.verts, mesh.normals,
                      mesh.indices, mesh.albedoTextures, mesh.normalTextures);
    } else {
        std::cerr << "Unsupported file format: " << filepath << std::endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string &jsonName) {
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // Volume integrator controls are scene-wide so diagnostic AOVs and the
    // reference/debug transport switch consume exactly the same stored fields.
    if (data.contains("Integrator")) {
        const auto &integrator = data["Integrator"];
        const std::string quality =
            integrator.value("VOLUME_QUALITY", "Reference");
        if (quality == "Reference") {
            volumeIntegrator.quality = VOLUME_QUALITY_REFERENCE;
        } else if (quality == "Debug") {
            volumeIntegrator.quality = VOLUME_QUALITY_DEBUG;
        } else {
            std::cerr << "Integrator VOLUME_QUALITY must be \"Reference\" or "
                         "\"Debug\"."
                      << std::endl;
            exit(-1);
        }
        volumeIntegrator.debugMode =
            parseVolumeDebugMode(integrator.value("VOLUME_DEBUG", "None"));
        volumeIntegrator.maxScatteringDepth =
            integrator.value("VOLUME_MAX_SCATTER_DEPTH", 8);
        volumeIntegrator.reportTrackingStats =
            integrator.value("VOLUME_STATS", true) ? 1 : 0;
        if (volumeIntegrator.maxScatteringDepth < 1) {
            std::cerr << "Integrator VOLUME_MAX_SCATTER_DEPTH must be >= 1."
                      << std::endl;
            exit(-1);
        }
    }

    // Reading materials
    const auto &materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;

    for (const auto &item : materialsData.items()) {
        // Here name must be unique for each material
        const auto &name = item.key();
        const auto &p = item.value();
        Material newMaterial{};

        if (p["TYPE"] == "Diffuse") {
            newMaterial.type = DIFFUSE;
            const auto &col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        } else if (p["TYPE"] == "Emitting") {
            const auto &col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            parseSpectrum(p, newMaterial.color, newMaterial.spectrumType,
                          newMaterial.blackbodyTemp, newMaterial.blackbodyNorm);
        } else if (p["TYPE"] == "Mirror") {
            if (!p.contains("SPEC_RGB")) {
                printf("You define a mirror material but you haven't specified "
                       "its "
                       "SPEC_RGB"
                       " property. The render will look wrong. \n");
                exit(-1);
            }

            newMaterial.type = MIRROR;
            const auto &spec_col = p["SPEC_RGB"];
            newMaterial.specularColor =
                glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
        } else if (p["TYPE"] == "Dielectric") {
            if (!p.contains("SPEC_RGB")) {
                printf("You define a dielectric material but you haven't "
                       "specified its "
                       "SPEC_RGB"
                       " property. The render will look wrong. \n");
                exit(-1);
            }

            if (!p.contains("IOR")) {
                printf("You define a dielectric material but you haven't "
                       "specified its "
                       "IOR"
                       " property. The render will look wrong. \n");
                exit(-1);
            }

            newMaterial.type = DIELECTRIC;
            const auto &spec_col = p["SPEC_RGB"];
            newMaterial.specularColor =
                glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
            // Optional dispersion (SPECTRAL builds): Abbe number V of the
            // glass; IOR is then interpreted as n_d (at the 587.6 nm d-line).
            // References: BK7 crown 64.17, SF11 flint 25.68, diamond 55.3.
            // 0 (absent) keeps the wavelength-independent behavior.
            if (p.contains("ABBE")) {
                newMaterial.abbe = (float)p["ABBE"];
            }
        } else if (p["TYPE"] == "Microfacet") {
            if (!p.contains("RGB")) {
                printf("You define a microfacet material but you haven't "
                       "specified its "
                       "RGB"
                       " property. The render will look wrong. \n");
                exit(-1);
            }

            if (!p.contains("SPEC_RGB")) {
                printf("You define a microfacet material but you haven't "
                       "specified its "
                       "SPEC_RGB"
                       " property. The render will look wrong. \n");
                exit(-1);
            }

            if (!p.contains("IOR")) {
                printf("You define a microfacet material but you haven't "
                       "specified its "
                       "IOR"
                       " property. The render will look wrong. \n");
                exit(-1);
            }

            if (!p.contains("ROUGHNESS")) {
                printf("You define a microfacet material but you haven't "
                       "specified its "
                       "ROUGHNESS"
                       " property. The render will look wrong. \n");
                exit(-1);
            }

            newMaterial.type = MICROFACET;
            const auto &col = p["RGB"];
            const auto &spec_col = p["SPEC_RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specularColor =
                glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.indexOfRefraction = p["IOR"];
        } else if (p["TYPE"] == "Medium") {
            // Participating medium: the geom's interior scatters/absorbs
            // light; its surface is an invisible null boundary. See the
            // Material struct and src/render/volume.h.
            //   SIGMA_A [r,g,b]: absorption per unit distance (default 0.1)
            //   SIGMA_S [r,g,b]: scattering per unit distance (default 1.0)
            //   G: Henyey-Greenstein asymmetry in (-1, 1) (default 0)
            //   DENSITY: global multiplier on both sigmas (default 1)
            //   HETEROGENEOUS: true -> fbm-noise density field (smoke/cloud)
            //   NOISE_SCALE / NOISE_OCTAVES: fbm frequency and octave count
            newMaterial.type = MEDIUM;
            newMaterial.sigmaA = glm::vec3(0.1f);
            newMaterial.sigmaS = glm::vec3(1.0f);
            newMaterial.sootSigmaA = glm::vec3(1.2f, 1.0f, 0.82f);
            newMaterial.sootSigmaS = glm::vec3(0.02f);
            newMaterial.flameAbsorptionScale = 0.04f;
            newMaterial.temperatureScale = 1.0f;
            newMaterial.sootEmission = 0.25f;
            newMaterial.mediumFieldModel = MEDIUM_FIELD_PROCEDURAL;
            if (p.contains("SIGMA_A")) {
                const auto &sa = p["SIGMA_A"];
                newMaterial.sigmaA = glm::vec3(sa[0], sa[1], sa[2]);
            }
            if (p.contains("SIGMA_S")) {
                const auto &ss = p["SIGMA_S"];
                newMaterial.sigmaS = glm::vec3(ss[0], ss[1], ss[2]);
            }
            newMaterial.hgG = p.value("G", 0.0f);
            newMaterial.densityScale = p.value("DENSITY", 1.0f);
            newMaterial.heterogeneous = p.value("HETEROGENEOUS", false) ? 1 : 0;
            newMaterial.noiseScale = p.value("NOISE_SCALE", 4.0f);
            newMaterial.noiseOctaves = p.value("NOISE_OCTAVES", 4);
            newMaterial.noiseSeed = p.value("NOISE_SEED", 0);

            if (p.contains("FIELD_MODEL")) {
                const std::string fieldModel = p["FIELD_MODEL"];
                if (fieldModel == "Procedural") {
                    newMaterial.mediumFieldModel = MEDIUM_FIELD_PROCEDURAL;
                } else if (fieldModel == "SparseGrid") {
                    newMaterial.mediumFieldModel = MEDIUM_FIELD_SPARSE_GRID;
                    newMaterial.heterogeneous = 1;
                } else {
                    std::cerr
                        << "Medium FIELD_MODEL must be \"Procedural\" or "
                           "\"SparseGrid\"."
                        << std::endl;
                    exit(-1);
                }
            }
            if (p.contains("SOOT_SIGMA_A")) {
                const auto &sa = p["SOOT_SIGMA_A"];
                newMaterial.sootSigmaA =
                    glm::vec3(sa[0], sa[1], sa[2]);
            }
            if (p.contains("SOOT_SIGMA_S")) {
                const auto &ss = p["SOOT_SIGMA_S"];
                newMaterial.sootSigmaS =
                    glm::vec3(ss[0], ss[1], ss[2]);
            }
            newMaterial.flameAbsorptionScale =
                p.value("FLAME_ABSORPTION_SCALE", 0.04f);
            newMaterial.temperatureScale =
                p.value("TEMPERATURE_SCALE", 1.0f);
            newMaterial.sootEmission =
                p.value("SOOT_EMISSION_MULTIPLIER", 0.25f);

            // Optional continuous volume emission. This is deliberately
            // separate from SIGMA_A/SIGMA_S: EMISSION is a source term per
            // unit distance, while the sigma values control attenuation.
            if (p.contains("EMISSION")) {
                newMaterial.emittance = p["EMISSION"];
                newMaterial.color = glm::vec3(1.0f);
                if (p.contains("EMISSION_RGB")) {
                    const auto &ec = p["EMISSION_RGB"];
                    newMaterial.color = glm::vec3(ec[0], ec[1], ec[2]);
                }
                parseSpectrum(p, newMaterial.color, newMaterial.spectrumType,
                              newMaterial.blackbodyTemp,
                              newMaterial.blackbodyNorm);
            }

            // Density-field shape (heterogeneous only): "Fbm" (default),
            // "Cloud" (cumulus billows), "Plume" (single rising smoke
            // column), "Flame" (tapered, forked fire), "Candle" (laminar
            // wick flame), "FireFront" (multi-source turbulent flames),
            // "SmokeFront" (merged convective smoke), or "Fog" (soft
            // atmospheric height layer).
            newMaterial.mediumProfile = MEDIUM_PROFILE_FBM;
            if (p.contains("PROFILE")) {
                const std::string prof = p["PROFILE"];
                if (prof == "Cloud") {
                    newMaterial.mediumProfile = MEDIUM_PROFILE_CLOUD;
                } else if (prof == "Plume") {
                    newMaterial.mediumProfile = MEDIUM_PROFILE_PLUME;
                } else if (prof == "Flame") {
                    newMaterial.mediumProfile = MEDIUM_PROFILE_FLAME;
                } else if (prof == "Candle") {
                    newMaterial.mediumProfile = MEDIUM_PROFILE_CANDLE;
                } else if (prof == "FireFront") {
                    newMaterial.mediumProfile = MEDIUM_PROFILE_FIRE_FRONT;
                } else if (prof == "SmokeFront") {
                    newMaterial.mediumProfile = MEDIUM_PROFILE_SMOKE_FRONT;
                } else if (prof == "Fog") {
                    newMaterial.mediumProfile = MEDIUM_PROFILE_FOG;
                } else if (prof != "Fbm") {
                    printf("Unknown medium PROFILE \"%s\" (expected \"Fbm\", "
                           "\"Cloud\", \"Plume\", \"Flame\", \"Candle\", "
                           "\"FireFront\", \"SmokeFront\", or \"Fog\").\n",
                           prof.c_str());
                    exit(-1);
                }
            }
            if (p.contains("PROFILE") &&
                newMaterial.mediumFieldModel == MEDIUM_FIELD_SPARSE_GRID) {
                std::cerr << "SparseGrid media use object VOLUME_GRID fields "
                             "and cannot also select a procedural PROFILE."
                          << std::endl;
                exit(-1);
            }
            if (p.contains("PROFILE") && !newMaterial.heterogeneous) {
                printf("Medium PROFILE requires HETEROGENEOUS=true; otherwise "
                       "the selected density field would be ignored.\n");
                exit(-1);
            }
            if (newMaterial.hgG <= -1.0f || newMaterial.hgG >= 1.0f) {
                printf("Medium G (Henyey-Greenstein asymmetry) must be in "
                       "(-1, 1).\n");
                exit(-1);
            }
            if (glm::any(glm::lessThan(newMaterial.sigmaA, glm::vec3(0.0f))) ||
                glm::any(glm::lessThan(newMaterial.sigmaS, glm::vec3(0.0f))) ||
                glm::any(glm::lessThan(newMaterial.sootSigmaA,
                                       glm::vec3(0.0f))) ||
                glm::any(glm::lessThan(newMaterial.sootSigmaS,
                                       glm::vec3(0.0f))) ||
                newMaterial.densityScale < 0.0f ||
                newMaterial.emittance < 0.0f ||
                newMaterial.flameAbsorptionScale < 0.0f ||
                newMaterial.temperatureScale <= 0.0f ||
                newMaterial.sootEmission < 0.0f) {
                printf("Medium optical coefficients, DENSITY, EMISSION, and "
                       "grid emission controls must be non-negative "
                       "(TEMPERATURE_SCALE must be positive).\n");
                exit(-1);
            }
            if (newMaterial.noiseScale <= 0.0f ||
                newMaterial.noiseOctaves < 1 ||
                newMaterial.noiseOctaves > 8) {
                printf("Medium NOISE_SCALE must be > 0 and NOISE_OCTAVES must "
                       "be in [1, 8].\n");
                exit(-1);
            }
#if SPECTRAL
            if (newMaterial.mediumFieldModel ==
                    MEDIUM_FIELD_SPARSE_GRID &&
                newMaterial.emittance > 0.0f &&
                newMaterial.spectrumType != SPECTRUM_BLACKBODY) {
                std::cerr
                    << "Emissive SparseGrid media require "
                       "SPECTRUM {\"BLACKBODY\": kelvin}; voxel temperature "
                       "is the primary color and HDR-power driver."
                    << std::endl;
                exit(-1);
            }
#endif
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }

    // Reading textures
    const auto &texturesData = data["Textures"];
    std::unordered_map<std::string, uint32_t> AlbedoTexToID;
    std::unordered_map<std::string, uint32_t> NormalTexToID;
    std::unordered_map<std::string, uint32_t> BumpTexToID;

    for (const auto &texture : texturesData.items()) {
        // Here name must be unique for each texture
        const auto &name = texture.key();
        const auto &p = texture.value();

        std::string textureType = p["TYPE"];
        if (textureType.empty()) {
            std::cerr << "You specify a texture but you haven't specify the "
                         "type of it."
                      << std::endl;
            exit(-1);
        } else if (textureType != "Albedo" && textureType != "Normal" &&
                   textureType != "Bump") {
            std::cerr << "Unsupported texture type: " << textureType
                      << std::endl;
            exit(-1);
        }

        if (!p.contains("TEXTURE_PATH")) {
            std::cerr << "No path provided for the texture. Cannot load."
                      << std::endl;
            exit(-1);
        }

        // Resolve the texture path relative to the scene JSON file rather than
        // the process working directory, so relative paths stay portable.
        std::filesystem::path texPath(p["TEXTURE_PATH"].get<std::string>());
        if (texPath.is_relative()) {
            texPath = std::filesystem::path(jsonName).parent_path() / texPath;
        }
        std::string filepath = texPath.string();
        glm::vec4 *curTexture;
        glm::ivec2 textureSize;
        loadTexture(filepath, textureType, curTexture, textureSize);

        if (p["TYPE"] == "Albedo") {
            AlbedoTexToID[name] = albedoTextures.size();
            albedoTextures.emplace_back(make_tuple(curTexture, textureSize));
            printf("Albedo texture added with ID: %d in the albedo texture "
                   "array. \n",
                   AlbedoTexToID[name]);
        } else if (p["TYPE"] == "Normal") {
            NormalTexToID[name] = normalTextures.size();
            normalTextures.emplace_back(make_tuple(curTexture, textureSize));
            printf("Normal texture added with ID: %d in the normal texture "
                   "array. \n",
                   NormalTexToID[name]);
        } else if (p["TYPE"] == "Bump") {
            BumpTexToID[name] = bumpTextures.size();
            bumpTextures.emplace_back(make_tuple(curTexture, textureSize));
            printf(
                "Bump texture added with ID: %d in the bump texture array. \n",
                BumpTexToID[name]);
        }
    }

    // Reading the optional environment map (equirectangular HDR). Escaped rays
    // sample this for both the background and image-based lighting.
    if (data.contains("Environment")) {
        const auto &env = data["Environment"];
        if (!env.contains("TEXTURE_PATH")) {
            std::cerr << "Environment block has no TEXTURE_PATH. Cannot load."
                      << std::endl;
            exit(-1);
        }

        // Resolve the path relative to the scene JSON file, like other
        // textures.
        std::filesystem::path envPath(env["TEXTURE_PATH"].get<std::string>());
        if (envPath.is_relative()) {
            envPath = std::filesystem::path(jsonName).parent_path() / envPath;
        }
        loadHDRTexture(envPath.string(), envMap, envMapSize);
        envMapIntensity =
            env.contains("INTENSITY") ? env["INTENSITY"].get<float>() : 1.0f;
        // ROTATION is authored in degrees (yaw around +Y); store radians.
        envMapRotation = env.contains("ROTATION")
                             ? env["ROTATION"].get<float>() * (PI / 180.0f)
                             : 0.0f;
        hasEnvMap = true;
        buildEnvDistribution();
        printf(
            "Environment map loaded (%dx%d, intensity %f, rotation %f deg)\n",
            envMapSize.x, envMapSize.y, envMapIntensity,
            envMapRotation * (180.0f / PI));
    }

    // Reading optional delta (point/directional) lights. These are singular
    // lights sampled only by next-event estimation (no geometry to hit).
    if (data.contains("Lights")) {
        for (const auto &l : data["Lights"]) {
            DeltaLight light{};
            const std::string ltype = l.value("TYPE", std::string("Point"));
            glm::vec3 rgb(1.0f);
            if (l.contains("RGB")) {
                rgb = glm::vec3(l["RGB"][0], l["RGB"][1], l["RGB"][2]);
            }
            parseSpectrum(l, rgb, light.spectrumType, light.blackbodyTemp,
                          light.blackbodyNorm);
            float intensity = l.value("INTENSITY", 1.0f);
            light.radiance = rgb * intensity;

            if (ltype == "Directional") {
                light.type = DIRECTIONAL_LIGHT;
                glm::vec3 dir(0.0f, -1.0f, 0.0f);
                if (l.contains("DIRECTION")) {
                    dir = glm::vec3(l["DIRECTION"][0], l["DIRECTION"][1],
                                    l["DIRECTION"][2]);
                }
                light.direction = glm::normalize(dir);
            } else { // Point
                light.type = POINT_LIGHT;
                if (l.contains("POSITION")) {
                    light.position = glm::vec3(
                        l["POSITION"][0], l["POSITION"][1], l["POSITION"][2]);
                }
            }
            deltaLights.push_back(light);
        }
        printf("Loaded %zu delta (point/directional) light(s)\n",
               deltaLights.size());
    }

    // Reading objects
    const auto &objectsData = data["Objects"];
    int numOfFaces = 0;
    std::vector<int> areaLightGeomIndices;
    for (const auto &p : objectsData) {
        const auto &type = p["TYPE"];
        const std::string &mat = p["MATERIAL"];

        Geom newGeom{};

        // Have to initialize the material IDs to -1, otherwise the default
        // value for int is 0 N we will have segmentation fault in CUDA
        newGeom.material.albedoTextureID = -1;
        newGeom.material.normalTextureID = -1;
        newGeom.material.bumpTextureID = -1;

        // Device pointers are filled in during upload (pathtraceInit).
        // Host-side mesh arrays live in newMeshData (empty for non-mesh geoms).
        newGeom.geometry.numTriangles = 0;
        newGeom.geometry.devTriangles = nullptr;
        newGeom.geometry.devNodes = nullptr;
        newGeom.volumeGridId = -1;
        newGeom.volumeGrid.valid = 0;
        MeshData newMeshData;

        if (type == "cube") {
            newGeom.type = CUBE;
        } else if (type == "sphere") {
            newGeom.type = SPHERE;
        } else if (type == "mesh") {
            newGeom.type = MESH;

            if (!p.contains("MESH_PATH")) {
                std::cerr << "No path provided for mesh object" << std::endl;
                exit(-1);
            }

            // MESH_PATH may be relative; resolve it against the directory of
            // the scene JSON file rather than the process working directory.
            std::filesystem::path meshPath(p["MESH_PATH"].get<std::string>());
            if (meshPath.is_relative()) {
                meshPath =
                    std::filesystem::path(jsonName).parent_path() / meshPath;
            }
            std::string filepath = meshPath.string();
            Mesh newMesh;
            loadMesh(filepath, newMesh);

            printf("Loaded mesh with %zu vertices, %zu normals, %zu faces, %zu "
                   "indices, %zu uvs\n",
                   newMesh.verts.size(), newMesh.normals.size(),
                   newMesh.faces.size(), newMesh.indices.size(),
                   newMesh.uvs.size());

            // Build the BVH. This reorders the mesh faces into bvh.triangles,
            // matching the order the linear BVH nodes index into.
            BVH bvh(newMesh.faces);

            size_t numTriangles = bvh.triangles.size();
            if (numTriangles == 0) {
                std::cerr << "No triangles found in mesh object" << std::endl;
                exit(-1);
            }

            newGeom.geometry.numTriangles = static_cast<int>(numTriangles);

            // Host-side arrays live in MeshData, owned by the Scene. We keep a
            // heap-allocated copy (freed with delete[]) that outlives the local
            // `bvh` (whose vector buffer is freed at the end of this scope).
            newMeshData.numTriangles = static_cast<int>(numTriangles);
            newMeshData.triangles = new Triangle[numTriangles];
            std::copy(bvh.triangles.begin(), bvh.triangles.end(),
                      newMeshData.triangles);

            // Heap-allocated copy of the flattened BVH nodes (same ownership
            // reasoning as the triangles above).
            newMeshData.numNodes = bvh.numNodes;
            newMeshData.nodes = new LinearBVHNode[bvh.numNodes];
            std::copy(bvh.nodes, bvh.nodes + bvh.numNodes, newMeshData.nodes);

            numOfFaces += numTriangles;

#if USE_SELF_LOADED_TEXTURES
            /** Here we are reading the textures if there are any **/
            if (!p.contains("TEXTURES")) {
                std::cerr << "No user added textures." << std::endl;
            } else {
                const auto &textures = p["TEXTURES"];
                for (const auto &texture : textures) {
                    std::string textureName = texture;
                    bool findAlbedo =
                        AlbedoTexToID.find(textureName) != AlbedoTexToID.end();
                    bool findNormal =
                        NormalTexToID.find(textureName) != NormalTexToID.end();
                    bool findBump =
                        BumpTexToID.find(textureName) != BumpTexToID.end();

                    if (!findAlbedo && !findNormal && !findBump) {
                        std::cerr << "Texture " << textureName
                                  << " not found in the scene" << std::endl;
                    }

                    newGeom.material.albedoTextureID =
                        findAlbedo ? AlbedoTexToID[textureName] : -1;
                    newGeom.material.normalTextureID =
                        findNormal ? NormalTexToID[textureName] : -1;
                    newGeom.material.bumpTextureID =
                        findBump ? BumpTexToID[textureName] : -1;
                }
            }
#else
            if (newMesh.albedoTextures.size() == 0) {
                printf("No albedo texture found for the mesh object. \n");
            } else {
                printf("%zu albedo texture found for the mesh object. \n",
                       newMesh.albedoTextures.size());
                for (const auto &texture : newMesh.albedoTextures) {
                    std::string textureName = std::get<0>(texture);
                    AlbedoTexToID[textureName] = albedoTextures.size();
                    albedoTextures.emplace_back(
                        make_tuple(std::get<1>(texture), std::get<2>(texture)));
                    newGeom.material.albedoTextureID =
                        AlbedoTexToID[textureName];
                }
            }

            if (newMesh.normalTextures.size() == 0) {
                printf("No normal texture found for the mesh object. \n");
            } else {
                printf("%zu normal texture found for the mesh object. \n",
                       newMesh.normalTextures.size());
                for (const auto &texture : newMesh.normalTextures) {
                    std::string textureName = std::get<0>(texture);
                    NormalTexToID[textureName] = normalTextures.size();
                    normalTextures.emplace_back(
                        make_tuple(std::get<1>(texture), std::get<2>(texture)));
                    newGeom.material.normalTextureID =
                        NormalTexToID[textureName];
                }
            }

            if (newMesh.bumpTextures.size() == 0) {
                printf("No bump texture found for the mesh object. \n");
            } else {
                printf("%zu bump texture found for the mesh object. \n",
                       newMesh.bumpTextures.size());
                for (const auto &texture : newMesh.bumpTextures) {
                    std::string textureName = std::get<0>(texture);
                    BumpTexToID[textureName] = bumpTextures.size();
                    bumpTextures.emplace_back(
                        make_tuple(std::get<1>(texture), std::get<2>(texture)));
                    newGeom.material.bumpTextureID = BumpTexToID[textureName];
                }
            }
#endif
        }

        newGeom.material.materialId = MatNameToID[mat];

        // Media need an analytic ray/interior interval for distance sampling
        // and shadow-ray transmittance (volume.h), which only the cube and
        // sphere primitives provide.
        if (materials[newGeom.material.materialId].type == MEDIUM &&
            newGeom.type == MESH) {
            std::cerr << "Medium materials are only supported on cube and "
                         "sphere objects (got a mesh)."
                      << std::endl;
            exit(-1);
        }

        const auto &trans = p["TRANS"];
        const auto &rotat = p["ROTAT"];
        const auto &scale = p["SCALE"];

        // translation/rotation/scale are only build-time inputs to the
        // transform matrices, so they stay local and are not stored on Geom
        // (the device never reads them).
        glm::vec3 translation(trans[0], trans[1], trans[2]);
        glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
        glm::vec3 scaling(scale[0], scale[1], scale[2]);
        newGeom.transform.transform = utilityCore::buildTransformationMatrix(
            translation, rotation, scaling);
        newGeom.transform.inverseTransform =
            glm::inverse(newGeom.transform.transform);
        newGeom.transform.invTranspose =
            glm::inverseTranspose(newGeom.transform.transform);
        newGeom.surfaceArea = geomSurfaceArea(newGeom, newMeshData);

        const Material &objectMaterial =
            materials[newGeom.material.materialId];
        if (p.contains("VOLUME_GRID")) {
            if (objectMaterial.type != MEDIUM ||
                objectMaterial.mediumFieldModel !=
                    MEDIUM_FIELD_SPARSE_GRID ||
                newGeom.type != CUBE) {
                std::cerr
                    << "VOLUME_GRID requires a cube using a Medium material "
                       "with FIELD_MODEL \"SparseGrid\"."
                    << std::endl;
                exit(-1);
            }
            const auto &vg = p["VOLUME_GRID"];
            if (!vg.is_object() || !vg.contains("PRESET")) {
                std::cerr << "VOLUME_GRID requires a PRESET (Candle, "
                             "WoodFire, or Wildfire)."
                          << std::endl;
                exit(-1);
            }
            const CombustionPreset preset =
                parseCombustionPreset(vg["PRESET"]);
            VolumeGridBuildSettings settings =
                defaultVolumeGridBuildSettings(preset);
            if (vg.contains("RESOLUTION")) {
                const auto &r = vg["RESOLUTION"];
                settings.cellResolution =
                    glm::ivec3(r[0], r[1], r[2]);
            }
            if (vg.contains("WIND")) {
                const auto &w = vg["WIND"];
                settings.wind = glm::vec3(w[0], w[1], w[2]);
            }
            settings.seed = vg.value("SEED", settings.seed);
            settings.activeThreshold =
                vg.value("ACTIVE_THRESHOLD", settings.activeThreshold);
            settings.densityScale =
                vg.value("DENSITY_SCALE", settings.densityScale);
            settings.sootScale =
                vg.value("SOOT_SCALE", settings.sootScale);
            settings.fuelScale =
                vg.value("FUEL_SCALE", settings.fuelScale);
            settings.temperatureScale =
                vg.value("FIELD_TEMPERATURE_SCALE",
                         settings.temperatureScale);
            settings.reactionScale =
                vg.value("REACTION_SCALE", settings.reactionScale);
            settings.turbulenceScale =
                vg.value("TURBULENCE_SCALE", settings.turbulenceScale);
            settings.buoyancyScale =
                vg.value("BUOYANCY_SCALE", settings.buoyancyScale);
            settings.smokeAdvection =
                vg.value("SMOKE_ADVECTION", settings.smokeAdvection);
            settings.sourceCompactness =
                vg.value("SOURCE_COMPACTNESS",
                         settings.sourceCompactness);
            settings.canopySpread =
                vg.value("CANOPY_SPREAD", settings.canopySpread);
            settings.canopyDensity =
                vg.value("CANOPY_DENSITY", settings.canopyDensity);

            printf("Building sparse combustion grid for geom %zu ...\n",
                   geoms.size());
            HostSparseVolumeGrid grid =
                buildSparseCombustionGrid(settings);
            std::string gridError;
            if (!validateSparseCombustionGrid(grid, &gridError)) {
                std::cerr << "Invalid generated VOLUME_GRID: " << gridError
                          << std::endl;
                exit(-1);
            }
            newGeom.volumeGridId = static_cast<int>(volumeGrids.size());
            printf("Sparse grid: %d x %d x %d cells, %zu active bricks "
                   "(%.1f MiB fields)\n",
                   grid.cellResolution.x, grid.cellResolution.y,
                   grid.cellResolution.z, grid.bricks.size(),
                   (grid.combustionSamples.size() +
                    grid.thermalFlowSamples.size()) *
                       sizeof(glm::vec4) / (1024.0 * 1024.0));
            printVolumeFieldSummary(grid);
            volumeGrids.emplace_back(std::move(grid));
        } else if (objectMaterial.type == MEDIUM &&
                   objectMaterial.mediumFieldModel ==
                       MEDIUM_FIELD_SPARSE_GRID) {
            std::cerr << "A SparseGrid medium object is missing VOLUME_GRID."
                      << std::endl;
            exit(-1);
        }

        geoms.push_back(newGeom);
        geomMeshData.push_back(newMeshData);
        // Register every emissive object as an area light so next-event
        // estimation can sample it. This must go by EMITTANCE, not by the
        // material being named "light": an emitter that NEE cannot sample
        // would still be MIS-down-weighted when a BSDF ray hits it (losing
        // energy), and multi-emitter scenes need distinct material names
        // (e.g. the per-spectrum light panels in the spectral demo scenes).
        if (materials[newGeom.material.materialId].emittance > 0.0f &&
            materials[newGeom.material.materialId].type != MEDIUM) {
            lights.push_back(newGeom);
            areaLightGeomIndices.push_back(
                static_cast<int>(geoms.size()) - 1);
            // Non-owning copy: shares geomMeshData's host pointers (freed once,
            // via geomMeshData).
            lightMeshData.push_back(newMeshData);
        }
    }

    bool hasEmissiveMedium = false;
    for (const Geom &g : geoms) {
        const Material &m = materials[g.material.materialId];
        hasEmissiveMedium |= m.type == MEDIUM && m.emittance > 0.0f;
    }
    if (lights.size() == 0 && !hasEnvMap && deltaLights.empty() &&
        !hasEmissiveMedium) {
        std::cerr
            << "No lights and no environment map found in the scene, your "
               "render will be pitch black!"
            << std::endl;
        exit(-1);
    }

    const auto &cameraData = data["Camera"];
    Camera &camera = state.camera;
    RenderState &state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto &pos = cameraData["EYE"];
    const auto &lookat = cameraData["LOOKAT"];
    const auto &up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // The path state currently assumes the camera begins in vacuum. Reject a
    // camera inside a medium explicitly instead of misclassifying the first
    // exit surface as an entry boundary.
    for (const Geom &g : geoms) {
        if (materials[g.material.materialId].type != MEDIUM) {
            continue;
        }
        glm::vec3 pl = glm::vec3(
            g.transform.inverseTransform * glm::vec4(camera.position, 1.0f));
        bool inside = g.type == SPHERE
                          ? glm::dot(pl, pl) < 0.25f
                          : glm::all(glm::lessThan(glm::abs(pl),
                                                   glm::vec3(0.5f)));
        if (inside) {
            std::cerr << "Camera starts inside a participating medium. This "
                         "renderer currently requires a vacuum camera; move "
                         "the camera or the medium boundary."
                      << std::endl;
            exit(-1);
        }
    }

    // PBRT's PowerLightSampler motivates this compact CDF. With only tens of
    // emitters, a binary-searched CDF is smaller than an alias table and has
    // negligible lookup cost. The same PMF is copied to the original geom for
    // reverse/hit-light MIS, which is required for an unbiased estimator.
    areaLightCdf.assign(lights.size(), 0.0f);
    std::vector<float> lightWeights(lights.size(), 0.0f);
    float totalLightWeight = 0.0f;
    for (std::size_t i = 0; i < lights.size(); ++i) {
        const Material &material =
            materials[lights[i].material.materialId];
        const glm::vec3 spectrumProxy =
            illuminantRGB(material.spectrumType, material.blackbodyTemp);
        const float emittedY = fmaxf(
            luminance(material.color * spectrumProxy) * material.emittance,
            0.0f);
        const float weight = lights[i].surfaceArea > 0.0f && emittedY > 0.0f
                                 ? fmaxf(lights[i].surfaceArea * emittedY,
                                         1e-8f)
                                 : 0.0f;
        lightWeights[i] = weight;
        totalLightWeight += weight;
    }
    if (!lights.empty() && totalLightWeight <= 0.0f) {
        std::fill(lightWeights.begin(), lightWeights.end(), 1.0f);
        totalLightWeight = static_cast<float>(lights.size());
    }
    float cumulativeLightPmf = 0.0f;
    for (std::size_t i = 0; i < lights.size(); ++i) {
        const float pmf = lightWeights[i] / totalLightWeight;
        cumulativeLightPmf += pmf;
        areaLightCdf[i] = cumulativeLightPmf;
        lights[i].areaLightSelectionPmf = pmf;
        geoms[areaLightGeomIndices[i]].areaLightSelectionPmf = pmf;
    }
    if (!areaLightCdf.empty()) {
        areaLightCdf.back() = 1.0f;
    }

    // Default to no depth of field. These MUST be written even when the JSON
    // omits them: Camera is a plain struct with no initializers, so leaving
    // them untouched reads uninitialized memory -- when that garbage happened
    // to be positive, generateRayFromCamera enabled DOF with an absurd lens
    // radius and every camera ray missed the scene (intermittent all-black
    // renders that came and went with unrelated heap-layout changes).
    camera.lensRadius = 0.0f;
    camera.focalDistance = 0.0f;
    camera.exposure = cameraData.value("EXPOSURE", 0.0f);
    camera.toneMap = cameraData.value("TONEMAP", false) ? 1 : 0;

    if (!cameraData.contains("LENS_RADIUS")) {
        printf("You haven't specified "
               "LENS_RADIUS"
               " for your camera. DOF will not work. \n");
    } else {
        camera.lensRadius = cameraData["LENS_RADIUS"];
        printf("Lens radius %f added to camera \n", camera.lensRadius);
    }

    if (!cameraData.contains("FOCAL_DISTANCE")) {
        printf("You haven't specified "
               "FOCAL_DISTANCE"
               " for your camera. DOF will not work. \n");
    } else {
        camera.focalDistance = cameraData["FOCAL_DISTANCE"];
        printf("Focal distance %f added to camera \n", camera.focalDistance);
    }

    // calculate fov based on resolution
    // FOVY is the full vertical field of view. The image-plane half-height is
    // tan(FOVY / 2), not tan(FOVY). The previous expression silently doubled
    // every authored field of view (45 degrees rendered as roughly 90).
    float yscaled = tan(0.5f * fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (2.0f * atan(xscaled) * 180) / PI;

    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    printf(
        "Scene loaded with %d triangles, %zu materials, %zu albedo textures, "
        "%zu normal textures, %zu bump textures, %zu objects, %zu lights\n",
        numOfFaces, materials.size(), albedoTextures.size(),
        normalTextures.size(), bumpTextures.size(), geoms.size(),
        lights.size());
}
