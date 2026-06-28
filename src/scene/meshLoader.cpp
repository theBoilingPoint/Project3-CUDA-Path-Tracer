#include "meshLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tinyobjloader/tiny_obj_loader.h"

#define TINYGLTF_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"

bool endsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.rfind(suffix) == (str.size() - suffix.size());
}

/**
 * @brief This function loads an OBJ file and stores the vertices, normals, and
 * UVs in the faces vector. The code is taken from the tinyobjloader repository:
 * https://github.com/tinyobjloader/tinyobjloader. Although the original
 * implementation can read meshes with arbitrarily-shaped faces, we are assuming
 * that the faces are triangles.
 *
 * @param filepath The absolute path to the OBJ file.
 */
void loadOBJ(const std::string &filepath, std::vector<Triangle> &faces,
             std::vector<glm::vec3> &verts, std::vector<glm::vec3> &normals,
             std::vector<glm::vec2> &uvs,
             std::vector<int> &indices) { // Pass by reference
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config)) {
        if (!reader.Error().empty()) {
            printf("TinyObjReader ERROR: %s\n", reader.Error().c_str());
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        printf("TinyObjReader WARNING: %s\n", reader.Warning().c_str());
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces (polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            if (fv != 3) {
                std::cerr
                    << "This OBJ loader only supports triangles. Exiting..."
                    << std::endl;
                exit(1);
            }

            std::vector<glm::vec3> verticesForOneFace;
            std::vector<glm::vec3> normalsForOneFace;
            std::vector<glm::vec2> uvsForOneFace;

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // Access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                glm::vec3 vertex(vx, vy, vz);
                verticesForOneFace.push_back(vertex);

                // Add vertex to verts vector
                verts.push_back(vertex);

                // Check if `normal_index` is zero or positive. negative = no
                // normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx =
                        attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny =
                        attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz =
                        attrib.normals[3 * size_t(idx.normal_index) + 2];
                    glm::vec3 normal(nx, ny, nz);
                    normalsForOneFace.push_back(normal);

                    // Add normal to normals vector
                    normals.push_back(normal);
                }

                // Add index to indices vector
                indices.push_back(idx.vertex_index);

                // Process texture coordinates if needed
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx =
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty =
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    ty = 1.0 - ty; // Flip Y-axis
                    uvsForOneFace.push_back(glm::vec2(tx, ty));

                    // Add UV to uvs vector
                    uvs.push_back(glm::vec2(tx, ty));
                }
            }

            // Create Triangle and populate normals and UVs if available
            Triangle t(verticesForOneFace[0], verticesForOneFace[1],
                       verticesForOneFace[2]);
            t.planeNormal = glm::normalize(
                glm::cross(verticesForOneFace[1] - verticesForOneFace[0],
                           verticesForOneFace[2] - verticesForOneFace[1]));

            if (!normalsForOneFace.empty()) {
                for (int i = 0; i < fv; i++) {
                    t.normals[i] = normalsForOneFace[i];
                }
            }
            if (!uvsForOneFace.empty()) {
                for (int i = 0; i < fv; i++) {
                    t.uvs[i] = uvsForOneFace[i];
                }
            }

            faces.push_back(t);

            index_offset += fv;
        }
    }
}

// Function to compute the transformation matrix for a node
glm::mat4 getNodeTransform(const tinygltf::Node &node) {
    glm::mat4 transform = glm::mat4(1.0f); // Identity matrix

    // Apply translation, if present
    if (!node.translation.empty()) {
        glm::vec3 translation = glm::vec3(
            node.translation[0], node.translation[1], node.translation[2]);
        transform = glm::translate(transform, translation);
    }

    // Apply rotation, if present
    if (!node.rotation.empty()) {
        glm::quat rotation =
            glm::quat(node.rotation[3], node.rotation[0], node.rotation[1],
                      node.rotation[2]); // Quaternion
        transform *= glm::mat4_cast(
            rotation); // Convert quaternion to matrix and multiply
    }

    // Apply scale, if present
    if (!node.scale.empty()) {
        glm::vec3 scale =
            glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        transform = glm::scale(transform, scale);
    }

    return transform;
}

// Decodes a float-backed accessor (glm::vec2 / glm::vec3) into a tightly packed
// vector, honoring the bufferView's byteStride (0 => tightly packed). This is
// what lets us read interleaved attribute buffers correctly.
template <typename T>
static std::vector<T> readFloatAccessor(const tinygltf::Model &model,
                                        const tinygltf::Accessor &accessor) {
    const tinygltf::BufferView &view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer &buffer = model.buffers[view.buffer];

    constexpr int numComponents = sizeof(T) / sizeof(float); // vec2->2, vec3->3
    // byteStride == 0 means the data is tightly packed for this accessor.
    const size_t stride = view.byteStride != 0
                              ? static_cast<size_t>(view.byteStride)
                              : numComponents * sizeof(float);
    const unsigned char *base =
        &buffer.data[view.byteOffset + accessor.byteOffset];

    std::vector<T> out(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        const float *p = reinterpret_cast<const float *>(base + i * stride);
        for (int c = 0; c < numComponents; ++c)
            out[i][c] = p[c];
    }
    return out;
}

// Helper function to extract and populate triangle data.
// Indices are pre-decoded to 32-bit (caller handles the glTF componentType),
// attributes are pre-decoded (stride handled), and only triangle-list topology
// is supported.
void populateTriangles(std::vector<Triangle> &faces,
                       std::vector<glm::vec3> &verts,
                       std::vector<glm::vec3> &norms, std::vector<int> &idxs,
                       const std::vector<uint32_t> &indices,
                       const std::vector<glm::vec3> &positions,
                       const std::vector<glm::vec3> &normals,
                       const std::vector<glm::vec2> &uvs,
                       const glm::mat4 &transform,
                       const glm::mat3 &normalMatrix) {
    const bool hasNormals = !normals.empty();
    const bool hasUVs = !uvs.empty();

    // Iterate over the indices in sets of 3 (triangles)
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        uint32_t idx0 = indices[i + 0];
        uint32_t idx1 = indices[i + 1];
        uint32_t idx2 = indices[i + 2];

        // Get vertex positions and apply transformation
        glm::vec3 p1 = glm::vec3(transform * glm::vec4(positions[idx0], 1.0f));
        glm::vec3 p2 = glm::vec3(transform * glm::vec4(positions[idx1], 1.0f));
        glm::vec3 p3 = glm::vec3(transform * glm::vec4(positions[idx2], 1.0f));

        // Get vertex normals, transformed by the inverse-transpose so they stay
        // correct under rotation/non-uniform scale. Fall back to the geometric
        // (world-space) normal when the mesh has no normals.
        glm::vec3 n1 = hasNormals
                           ? glm::normalize(normalMatrix * normals[idx0])
                           : glm::normalize(glm::cross(p2 - p1, p3 - p2));
        glm::vec3 n2 =
            hasNormals ? glm::normalize(normalMatrix * normals[idx1]) : n1;
        glm::vec3 n3 =
            hasNormals ? glm::normalize(normalMatrix * normals[idx2]) : n1;

        // Get vertex UVs
        glm::vec2 uv1 = hasUVs ? uvs[idx0] : glm::vec2(0.0f);
        glm::vec2 uv2 = hasUVs ? uvs[idx1] : glm::vec2(0.0f);
        glm::vec2 uv3 = hasUVs ? uvs[idx2] : glm::vec2(0.0f);

        // Create Triangle and populate faces vector
        Triangle tri;
        uint32_t idx[3] = {idx0, idx1, idx2};
        glm::vec3 points[3] = {p1, p2, p3};
        glm::vec3 normalsArr[3] = {n1, n2, n3};
        glm::vec2 uvsArr[3] = {uv1, uv2, uv3};

        for (int i = 0; i < 3; ++i) {
            tri.points[i] = points[i];
            tri.normals[i] = normalsArr[i];
            tri.uvs[i] = uvsArr[i];

            // Add vertex to verts vector
            idxs.push_back(idx[i]);
            verts.push_back(points[i]);
            norms.push_back(normalsArr[i]);
        }

        tri.planeNormal = glm::normalize(glm::cross(p2 - p1, p3 - p2));

        faces.push_back(tri);
    }
}

// Recursive function to traverse nodes and extract mesh data
void extractMeshDataFromGLTF(const tinygltf::Model &model, int nodeIndex,
                             std::vector<Triangle> &faces,
                             std::vector<glm::vec3> &verts,
                             std::vector<glm::vec3> &norms,
                             std::vector<int> &idxs,
                             const glm::mat4 &parentTransform) {
    const tinygltf::Node &node = model.nodes[nodeIndex];

    // Compute the transformation matrix for this node
    glm::mat4 nodeTransform = parentTransform * getNodeTransform(node);

    // If the node contains a mesh, process it
    if (node.mesh >= 0) {
        const tinygltf::Mesh &mesh = model.meshes[node.mesh];

        for (const auto &primitive : mesh.primitives) {
            // Get POSITION attribute (vertex positions)
            const auto posIt = primitive.attributes.find("POSITION");
            if (posIt == primitive.attributes.end()) {
                std::cerr << "No POSITION attribute found" << std::endl;
                continue;
            }
            const tinygltf::Accessor &posAccessor =
                model.accessors[posIt->second];
            std::vector<glm::vec3> positions =
                readFloatAccessor<glm::vec3>(model, posAccessor);

            // Get NORMAL attribute (vertex normals), if available
            std::vector<glm::vec3> normals;
            const auto normIt = primitive.attributes.find("NORMAL");
            if (normIt != primitive.attributes.end()) {
                normals = readFloatAccessor<glm::vec3>(
                    model, model.accessors[normIt->second]);
            }

            // Get TEXCOORD_0 attribute (UVs), if available
            std::vector<glm::vec2> uvs;
            const auto uvIt = primitive.attributes.find("TEXCOORD_0");
            if (uvIt != primitive.attributes.end()) {
                uvs = readFloatAccessor<glm::vec2>(
                    model, model.accessors[uvIt->second]);
            }

            // Access indices if they exist.
            if (primitive.indices >= 0) {
                // We only support triangle lists. glTF's default mode (when
                // unspecified, tinygltf reports -1) is TRIANGLES per the spec.
                int mode = primitive.mode < 0 ? TINYGLTF_MODE_TRIANGLES
                                              : primitive.mode;
                if (mode != TINYGLTF_MODE_TRIANGLES) {
                    std::cerr << "Only triangle-list primitives are supported. "
                                 "Skipping primitive with mode: "
                              << primitive.mode << std::endl;
                    continue;
                }

                const tinygltf::Accessor &indexAccessor =
                    model.accessors[primitive.indices];
                const tinygltf::BufferView &indexBufferView =
                    model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer &indexBuffer =
                    model.buffers[indexBufferView.buffer];
                const unsigned char *indexBase =
                    &indexBuffer.data[indexBufferView.byteOffset +
                                      indexAccessor.byteOffset];

                // Decode indices to 32-bit regardless of the source
                // componentType (UNSIGNED_BYTE/SHORT/INT are all valid).
                std::vector<uint32_t> indices(indexAccessor.count);
                switch (indexAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    const uint8_t *p =
                        reinterpret_cast<const uint8_t *>(indexBase);
                    for (size_t i = 0; i < indexAccessor.count; ++i)
                        indices[i] = p[i];
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const uint16_t *p =
                        reinterpret_cast<const uint16_t *>(indexBase);
                    for (size_t i = 0; i < indexAccessor.count; ++i)
                        indices[i] = p[i];
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const uint32_t *p =
                        reinterpret_cast<const uint32_t *>(indexBase);
                    for (size_t i = 0; i < indexAccessor.count; ++i)
                        indices[i] = p[i];
                    break;
                }
                default:
                    std::cerr << "Unsupported index component type: "
                              << indexAccessor.componentType << std::endl;
                    continue;
                }

                // Normals must be transformed by the inverse-transpose of the
                // node transform so they survive rotation/non-uniform scale.
                glm::mat3 normalMatrix =
                    glm::transpose(glm::inverse(glm::mat3(nodeTransform)));

                populateTriangles(faces, verts, norms, idxs, indices, positions,
                                  normals, uvs, nodeTransform, normalMatrix);
            }
        }
    }

    // Recursively traverse child nodes
    for (size_t i = 0; i < node.children.size(); ++i) {
        extractMeshDataFromGLTF(model, node.children[i], faces, verts, norms,
                                idxs, nodeTransform);
    }
}

// Entry function to traverse the GLTF scene
void extractMeshDataFromGLTFScene(const tinygltf::Model &model,
                                  std::vector<Triangle> &faces,
                                  std::vector<glm::vec3> &verts,
                                  std::vector<glm::vec3> &normals,
                                  std::vector<int> &indices) {
    const glm::mat4 identityMatrix = glm::mat4(1.0f); // Identity matrix

    // Start with the root nodes in the default scene
    const tinygltf::Scene &scene = model.scenes[model.defaultScene];
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        extractMeshDataFromGLTF(model, scene.nodes[i], faces, verts, normals,
                                indices, identityMatrix);
    }
}

void extractTextureFromGLTFScene(
    const tinygltf::Image &image, TextureType type, const std::string &name,
    std::vector<std::tuple<std::string, glm::vec4 *, glm::ivec2>>
        &albedoTextures,
    std::vector<std::tuple<std::string, glm::vec4 *, glm::ivec2>>
        &normalTextures) {
    int width = image.width;
    int height = image.height;
    const unsigned char *imageData = image.image.data();

    if (!imageData) {
        std::cerr << "No image data found for texture: " << name << std::endl;
        return;
    }

    // Dynamically allocate an array of glm::vec4 to hold texture data
    glm::vec4 *textureData = new glm::vec4[width * height];

    switch (type) {
    case TextureType::ALBEDO:
        for (int i = 0; i < width * height; ++i) {
            int pixelIndex = i * 4; // Assuming RGBA, 4 bytes per pixel

            // Convert the raw image data (unsigned char) to floating-point [0,
            // 1] glm::vec4
            textureData[i] =
                glm::vec4(imageData[pixelIndex] / 255.0f,     // Red
                          imageData[pixelIndex + 1] / 255.0f, // Green
                          imageData[pixelIndex + 2] / 255.0f, // Blue
                          imageData[pixelIndex + 3] / 255.0f  // Alpha
                );
        }
        albedoTextures.push_back(
            std::make_tuple(name, textureData, glm::ivec2(width, height)));
        printf("Albedo map for material: %s loaded from the GLTF/GLB scene "
               "with dimesnions: %d x %d\n",
               name.c_str(), width, height);
        break;
    case TextureType::NORMAL:
        for (int i = 0; i < width * height; ++i) {
            int pixelIndex = i * 4; // Assuming RGBA, 4 bytes per pixel

            // Convert the raw image data (unsigned char) to normal map data
            // Normal maps usually store values in the [0, 255] range for X and
            // Y and [0, 1] for Z (blue)
            float nx = (imageData[pixelIndex] / 255.0f) * 2.0f -
                       1.0f; // Red channel for X, range [-1, 1]
            float ny = (imageData[pixelIndex + 1] / 255.0f) * 2.0f -
                       1.0f; // Green channel for Y, range [-1, 1]
            float nz = (imageData[pixelIndex + 2] /
                        255.0f); // Blue channel for Z, range [0, 1]
            float alpha =
                imageData[pixelIndex + 3] /
                255.0f; // Alpha channel (not often used in normal maps)

            // Store the normal vector in the glm::vec4 (with w representing
            // alpha)
            textureData[i] = glm::vec4(nx, ny, nz, alpha);
        }
        normalTextures.push_back(
            std::make_tuple(name, textureData, glm::ivec2(width, height)));
        printf("Normal map for material: %s loaded from the GLTF/GLB scene "
               "with dimesnions: %d x %d\n",
               name.c_str(), width, height);
        break;
    default:
        std::cerr << "Unsupported texture type." << std::endl;
        return;
    }
}

void loadGLTFTexture(
    const tinygltf::Model &model,
    std::vector<std::tuple<std::string, glm::vec4 *, glm::ivec2>>
        &albedoTextures,
    std::vector<std::tuple<std::string, glm::vec4 *, glm::ivec2>>
        &normalTextures) {
    // Loop over each material
    for (const auto &material : model.materials) {
        const std::string matName = material.name;
        std::cout << "Material: " << matName << std::endl;

        // Base Color (Albedo)
        if (material.values.find("baseColorTexture") != material.values.end()) {
            int textureIndex =
                material.values.at("baseColorTexture").TextureIndex();
            const tinygltf::Texture &texture = model.textures.at(textureIndex);

            if (texture.source >= 0 && texture.source < model.images.size()) {
                const tinygltf::Image &image = model.images.at(texture.source);
                extractTextureFromGLTFScene(image, TextureType::ALBEDO, matName,
                                            albedoTextures, normalTextures);
            } else {
                std::cout
                    << "No valid image source found for texture of material: "
                    << matName << " at URL: " << texture.source << std::endl;
                return;
            }
        } else {
            std::cout << "No albedo texture found for material: " << matName
                      << std::endl;
        }

        // Normal Map
        if (material.additionalValues.find("normalTexture") !=
            material.additionalValues.end()) {
            int textureIndex =
                material.additionalValues.at("normalTexture").TextureIndex();
            const tinygltf::Texture &texture = model.textures.at(textureIndex);

            if (texture.source >= 0 && texture.source < model.images.size()) {
                const tinygltf::Image &image = model.images.at(texture.source);
                extractTextureFromGLTFScene(image, TextureType::NORMAL, matName,
                                            albedoTextures, normalTextures);
            } else {
                std::cout << "No valid image source found for normal map of "
                             "material: "
                          << matName << std::endl;
                return;
            }
        } else {
            std::cout << "No normal map found for material: " << matName
                      << " at URL: " << std::endl;
        }

        printf("\n");
    }
}

void loadGLTFOrGLB(const std::string &filepath, std::vector<Triangle> &faces,
                   std::vector<glm::vec3> &verts,
                   std::vector<glm::vec3> &normals, std::vector<int> &indices,
                   std::vector<std::tuple<std::string, glm::vec4 *, glm::ivec2>>
                       &albedoTextures,
                   std::vector<std::tuple<std::string, glm::vec4 *, glm::ivec2>>
                       &normalTextures) {

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
    if (endsWith(filepath, ".gltf")) {
        printf("Loading GLTF file: %s\n", filepath.c_str());
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
    } else if (endsWith(filepath, ".glb")) {
        printf("Loading GLB file: %s\n", filepath.c_str());
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
    } else {
        printf("Unsupported file extension (expected .gltf or .glb): %s\n",
               filepath.c_str());
    }

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF\n");
        exit(-1);
    }

    extractMeshDataFromGLTFScene(model, faces, verts, normals, indices);
    loadGLTFTexture(model, albedoTextures, normalTextures);
}

void loadTexture(const std::string &filepath, const std::string &textureType,
                 glm::vec4 *&texture, glm::ivec2 &textureSize) {
    int width, height, channels;
    unsigned char *imageData =
        stbi_load(filepath.c_str(), &width, &height, &channels,
                  STBI_rgb_alpha); // Force RGBA

    if (!imageData) {
        std::cerr << "Failed to load texture map: " << filepath
                  << ". Please check if your file path is correct and if the "
                     "file type is supported by stbi_load: .jpeg, .jpg, .png, "
                     ".tga, .bmp, .psd, .gif, .hdr, .pic, .pgm. \n"
                  << std::endl;
        exit(-1);
    }

    // Dynamically allocate an array of glm::vec4 to hold texture data
    texture = new glm::vec4[width * height];
    textureSize = glm::ivec2(width, height);

    if (textureType == "Albedo") {
        for (int i = 0; i < width * height; ++i) {
            int pixelIndex = i * 4; // 4 bytes per pixel (RGBA)

            // Store the image data as glm::vec4 (normalized to [0, 1] range)
            texture[i] = glm::vec4(imageData[pixelIndex] / 255.0f,     // Red
                                   imageData[pixelIndex + 1] / 255.0f, // Green
                                   imageData[pixelIndex + 2] / 255.0f, // Blue
                                   imageData[pixelIndex + 3] / 255.0f  // Alpha
            );
        }
    } else if (textureType == "Normal") {
        for (int i = 0; i < width * height; ++i) {
            int pixelIndex = i * 4; // 4 bytes per pixel (RGBA)

            // Convert normal data (Red and Green go from [0, 255] to [-1, 1])
            float nx = (imageData[pixelIndex] / 255.0f) * 2.0f -
                       1.0f; // Red channel (X direction)
            float ny = (imageData[pixelIndex + 1] / 255.0f) * 2.0f -
                       1.0f; // Green channel (Y direction)
            float nz = (imageData[pixelIndex + 2] /
                        255.0f); // Blue channel (Z direction)
            float alpha = imageData[pixelIndex + 3] /
                          255.0f; // Alpha (unused, but kept for compatibility)

            // Store the converted normal vector
            texture[i] = glm::vec4(nx, ny, nz, alpha);
        }
    } else if (textureType == "Bump") {
        for (int i = 0; i < width * height; ++i) {
            // imageData is RGBA (4 bytes/pixel); read the red channel of each
            // pixel as the grayscale height.
            float heightValue =
                imageData[i * 4] /
                255.0f; // Normalize the grayscale value to [0, 1]

            // Store the grayscale value in the RGB components (and alpha
            // as 1.0f)
            texture[i] = glm::vec4(heightValue, heightValue, heightValue, 1.0f);
        }
    } else {
        std::cerr << "Unsupported texture type: " << textureType << std::endl;
        return;
    }

    stbi_image_free(imageData);
}