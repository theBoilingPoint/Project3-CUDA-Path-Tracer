#include "main.h"
#include "color.h"
#include "preview.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <vector>

#ifdef USE_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

// Force NVIDIA GPU on Optimus / hybrid-graphics laptops so that the OpenGL
// context and the CUDA device are on the same physical GPU.
extern "C" { __declspec(dllexport) unsigned long NvOptimusEnablement = 1; }
extern "C" { __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1; }

static std::string startTimeString;
static bool headlessMode = false;
static bool denoiseMode = false;
static std::string outputBaseOverride;
static int sppOverride = 0;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json [--headless] [--denoise] [--spp N] "
               "[--output FILE_BASE]\n",
               argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--headless") == 0) {
            headlessMode = true;
        } else if (strcmp(argv[i], "--denoise") == 0) {
            denoiseMode = true;
        } else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) {
            sppOverride = atoi(argv[++i]);
            if (sppOverride <= 0) {
                fprintf(stderr, "--spp must be a positive integer.\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            outputBaseOverride = argv[++i];
            if (outputBaseOverride.size() >= 4 &&
                outputBaseOverride.substr(outputBaseOverride.size() - 4) ==
                    ".png") {
                outputBaseOverride.resize(outputBaseOverride.size() - 4);
            }
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return 1;
        }
    }

    // Load scene file
    scene = new Scene(sceneFile);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;
    if (sppOverride > 0) {
        renderState->iterations = (unsigned int)sppOverride;
    }

    if (headlessMode) {
        cudaError_t deviceErr = cudaSetDevice(0);
        if (deviceErr != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice(0) failed: %s\n",
                    cudaGetErrorString(deviceErr));
            return 1;
        }

        printf("Headless render: %dx%d, %u spp, depth %d\n", width, height,
               renderState->iterations, renderState->traceDepth);
        const auto begin = std::chrono::steady_clock::now();
        pathtraceInit(scene);
        for (iteration = 1; iteration <= (int)renderState->iterations;
             ++iteration) {
            pathtrace(nullptr, 0, iteration);
            if (iteration == 1 || iteration % 25 == 0 ||
                iteration == (int)renderState->iterations) {
                printf("  %d / %u spp\n", iteration, renderState->iterations);
            }
        }
        // The for-loop increment leaves `iteration` at N + 1; saveImage uses
        // it as the accumulation divisor, so restore the actual sample count.
        iteration = (int)renderState->iterations;
        bool saved = saveImage();
        pathtraceFree();
        const auto end = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(end - begin).count();
        printf("Headless render completed in %.2f s (%.3f s/spp).\n", seconds,
               seconds / (double)renderState->iterations);
        delete guiData;
        delete scene;
        cudaDeviceReset();
        return saved ? 0 : 1;
    }

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    // theta parameterizes the camera POSITION direction around lookAt (see the
    // camchanged block in runCuda: position.y = zoom * cos(theta) + lookAt.y),
    // which points OPPOSITE to the view vector. Deriving it from `view` flips
    // the camera below the target for downward-pitched cameras (every EYE/
    // LOOKAT pair with differing heights); use -view.
    theta = glm::acos(glm::dot(glm::normalize(-viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    return 0;
}

#ifdef USE_OIDN
static bool denoiseLinearImage(std::vector<glm::vec3> &pixels) {
    const size_t pixelCount = pixels.size();
    const size_t floatCount = 3 * pixelCount;
    const size_t byteCount = floatCount * sizeof(float);
    std::vector<float> input(floatCount);
    std::vector<float> output(floatCount);
    for (size_t i = 0; i < pixelCount; ++i) {
        input[3 * i + 0] = fmaxf(pixels[i].x, 0.0f);
        input[3 * i + 1] = fmaxf(pixels[i].y, 0.0f);
        input[3 * i + 2] = fmaxf(pixels[i].z, 0.0f);
    }

    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CUDA);
    if (!device) {
        fprintf(stderr, "OIDN CUDA device creation failed; saving raw PNG.\n");
        return false;
    }
    device.commit();

    oidn::BufferRef inputBuffer = device.newBuffer(byteCount);
    oidn::BufferRef outputBuffer = device.newBuffer(byteCount);
    inputBuffer.write(0, byteCount, input.data());

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", inputBuffer, oidn::Format::Float3, width, height);
    filter.setImage("output", outputBuffer, oidn::Format::Float3, width, height);
    filter.set("hdr", true);
    filter.set("quality", oidn::Quality::High);
    filter.commit();
    filter.execute();
    device.sync();

    const char *errorMessage = nullptr;
    if (device.getError(errorMessage) != oidn::Error::None) {
        fprintf(stderr, "OIDN failed: %s; saving raw PNG.\n",
                errorMessage ? errorMessage : "unknown error");
        return false;
    }

    outputBuffer.read(0, byteCount, output.data());
    for (size_t i = 0; i < pixelCount; ++i) {
        pixels[i] = glm::max(
            glm::vec3(output[3 * i + 0], output[3 * i + 1],
                      output[3 * i + 2]),
            glm::vec3(0.0f));
    }
    return true;
}
#endif

bool saveImage()
{
    if (iteration <= 0) {
        fprintf(stderr, "Cannot save before at least one sample is rendered.\n");
        return false;
    }
    float samples = iteration;
    // output image file
    Image img(width, height);
    Image rawPreview(width, height);
    Image linearImg(width, height);
    std::vector<glm::vec3> rawPixels(width * height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            rawPixels[index] = renderState->image[index] / samples;
        }
    }

    std::vector<glm::vec3> displayPixels = rawPixels;
    if (denoiseMode) {
#ifdef USE_OIDN
        if (denoiseLinearImage(displayPixels)) {
            printf("Denoised display PNG with OIDN CUDA (raw HDR preserved).\n");
        }
#else
        fprintf(stderr,
                "--denoise requested, but this build does not include OIDN.\n");
#endif
    }

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            linearImg.setPixel(width - 1 - x, y, rawPixels[index]);
            rawPreview.setPixel(
                width - 1 - x, y,
                displayTransform(rawPixels[index],
                                 renderState->camera.exposure,
                                 renderState->camera.toneMap));
            glm::vec3 pix =
                displayTransform(displayPixels[index],
                                 renderState->camera.exposure,
                                 renderState->camera.toneMap);
            img.setPixel(width - 1 - x, y, pix);
        }
    }

    std::string filename = outputBaseOverride;
    if (filename.empty()) {
        filename = renderState->imageName;
        std::ostringstream ss;
        ss << filename << "." << startTimeString << "." << samples << "samp";
        filename = ss.str();
    }

    std::filesystem::path outputPath(filename);
    if (outputPath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(outputPath.parent_path(), ec);
        if (ec) {
            fprintf(stderr, "Failed to create output directory \"%s\": %s\n",
                    outputPath.parent_path().string().c_str(),
                    ec.message().c_str());
            return false;
        }
    }

    bool saved = img.savePNG(filename);
    if (denoiseMode) {
        saved = rawPreview.savePNG(filename + ".raw") && saved;
    }
    if (headlessMode) {
        saved = linearImg.saveHDR(filename) && saved;
    }
    return saved;
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);

        // Define your own save interval
        if (iteration % 50 == 0) {
            saveImage();
        }
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
	        case GLFW_KEY_ESCAPE:
	            saveImage();
	            glfwSetWindowShouldClose(window, GL_TRUE);
	            break;
	        case GLFW_KEY_S:
	            saveImage();
	            break;
	        case GLFW_KEY_SPACE:
	            camchanged = true;
	            renderState = &scene->state;
	            Camera& cam = renderState->camera;
	            cam.lookAt = ogLookAt;
	            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
    	return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
