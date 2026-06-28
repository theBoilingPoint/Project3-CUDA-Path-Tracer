#include "scene.h"

#include <glm/gtc/matrix_inverse.hpp>

#define USE_SELF_LOADED_TEXTURES 1

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

        // Resolve the path relative to the scene JSON file, like other textures.
        std::filesystem::path envPath(env["TEXTURE_PATH"].get<std::string>());
        if (envPath.is_relative()) {
            envPath = std::filesystem::path(jsonName).parent_path() / envPath;
        }
        loadHDRTexture(envPath.string(), envMap, envMapSize);
        envMapIntensity = env.contains("INTENSITY")
                              ? env["INTENSITY"].get<float>()
                              : 1.0f;
        // ROTATION is authored in degrees (yaw around +Y); store radians.
        envMapRotation = env.contains("ROTATION")
                             ? env["ROTATION"].get<float>() * (PI / 180.0f)
                             : 0.0f;
        hasEnvMap = true;
        printf("Environment map loaded (%dx%d, intensity %f, rotation %f deg)\n",
               envMapSize.x, envMapSize.y, envMapIntensity,
               envMapRotation * (180.0f / PI));
    }

    // Reading objects
    const auto &objectsData = data["Objects"];
    int numOfFaces = 0;
    for (const auto &p : objectsData) {
        const auto &type = p["TYPE"];
        const std::string &mat = p["MATERIAL"];

        Geom newGeom;

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

        geoms.push_back(newGeom);
        geomMeshData.push_back(newMeshData);
        if (mat == "light") {
            lights.push_back(newGeom);
            // Non-owning copy: shares geomMeshData's host pointers (freed once,
            // via geomMeshData).
            lightMeshData.push_back(newMeshData);
        }
    }

    if (lights.size() == 0 && !hasEnvMap) {
        std::cerr << "No lights and no environment map found in the scene, your "
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
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;

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
