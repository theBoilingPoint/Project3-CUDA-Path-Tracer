#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <stb_image_write.h>

#include "image.h"

Image::Image(int x, int y)
    : xSize(x), ySize(y), pixels(new glm::vec3[x * y]) 
{}

Image::~Image()
{
    delete[] pixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel)
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

bool Image::savePNG(const std::string &baseFilename)
{
    std::vector<unsigned char> bytes(3 * xSize * ySize);
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char)pix.x;
            bytes[3 * i + 1] = (unsigned char)pix.y;
            bytes[3 * i + 2] = (unsigned char)pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    if (!stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes.data(),
                        xSize * 3)) {
        std::cerr << "Failed to save " << filename << "." << std::endl;
        return false;
    }
    std::cout << "Saved " << filename << "." << std::endl;
    return true;
}

bool Image::saveHDR(const std::string &baseFilename)
{
    // Radiance RGBE cannot represent signed RGB. Spectral XYZ-to-sRGB
    // conversion can produce small negative out-of-gamut components, so store
    // a nonnegative scene-linear image rather than handing invalid values to
    // stb's RGBE encoder.
    std::vector<float> linear(3 * xSize * ySize);
    for (int i = 0; i < xSize * ySize; ++i) {
        for (int c = 0; c < 3; ++c) {
            float value = pixels[i][c];
            linear[3 * i + c] =
                std::isfinite(value) ? std::max(value, 0.0f) : 0.0f;
        }
    }
    std::string filename = baseFilename + ".hdr";
    if (!stbi_write_hdr(filename.c_str(), xSize, ySize, 3, linear.data())) {
        std::cerr << "Failed to save " << filename << "." << std::endl;
        return false;
    }
    std::cout << "Saved " + filename + "." << std::endl;
    return true;
}
