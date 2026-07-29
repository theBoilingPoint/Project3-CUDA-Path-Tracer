#pragma once

#include <glm/glm.hpp>

using namespace std;

class Image
{
private:
    int xSize;
    int ySize;
    glm::vec3 *pixels;

public:
    Image(int x, int y);
    ~Image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    bool savePNG(const std::string &baseFilename);
    bool saveHDR(const std::string &baseFilename);
};
