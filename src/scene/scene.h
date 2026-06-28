#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "glm/glm.hpp"
#include <nlohmann/json.hpp>

#include "bvh.h"
#include "meshLoader.h"
#include "sceneStructs.h"
#include "utilities.h"

using namespace std;
using json = nlohmann::json;

class Mesh {
  public:
    Mesh();
    ~Mesh();

    vector<Triangle> faces;
    vector<int> indices;
    vector<glm::vec3> verts;
    vector<glm::vec3> normals;
    vector<glm::vec2> uvs;

    // The glm::vec4* in each of this vars is deleted in pathrace.cu once they
    // are loaded to the GPU
    vector<tuple<string, glm::vec4 *, glm::ivec2>> albedoTextures;
    vector<tuple<string, glm::vec4 *, glm::ivec2>> normalTextures;
    vector<tuple<string, glm::vec4 *, glm::ivec2>> bumpTextures;
};

class Scene {
  private:
    ifstream fp_in;

    void loadMesh(const string &filepath, Mesh &mesh);
    void loadFromJSON(const string &jsonName);
    template <typename T>
    void getValueFromJson(const json &data, const string &key, T &value);

  public:
    Scene(string filename);
    ~Scene();

    vector<Geom> geoms;
    vector<Geom> lights;
    // Host-side mesh arrays, parallel to `geoms`/`lights`. `geomMeshData` owns
    // the arrays; `lightMeshData` holds non-owning copies (lights are copies of
    // geoms), so only `geomMeshData` is freed.
    vector<MeshData> geomMeshData;
    vector<MeshData> lightMeshData;
    vector<Material> materials;
    vector<tuple<glm::vec4 *, glm::ivec2>> albedoTextures;
    vector<tuple<glm::vec4 *, glm::ivec2>> normalTextures;
    vector<tuple<glm::vec4 *, glm::ivec2>> bumpTextures;

    // Optional equirectangular HDR environment map. `envMap` is owned host-side
    // CPU pixel data (freed in the destructor); it is uploaded to a CUDA texture
    // object during deviceSceneInit. `hasEnvMap` is false when none is set.
    glm::vec4 *envMap = nullptr;
    glm::ivec2 envMapSize = glm::ivec2(0);
    float envMapIntensity = 1.0f;
    float envMapRotation = 0.0f; // Yaw around +Y, in radians
    bool hasEnvMap = false;

    RenderState state;
};
