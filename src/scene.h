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
    vector<Material> materials;
    vector<tuple<glm::vec4 *, glm::ivec2>> albedoTextures;
    vector<tuple<glm::vec4 *, glm::ivec2>> normalTextures;
    vector<tuple<glm::vec4 *, glm::ivec2>> bumpTextures;
    RenderState state;
};
