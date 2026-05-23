#include "stb_image.h"
#include <string>
#include <vector>

std::vector<unsigned char> process_image(const std::string& filepath, int& width, int& height){
    int channels;
    unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &channels, 3);
    if(data == nullptr){
        return {};
    }

    std::vector<unsigned char> img_data(data, data+ width * height * 3);

    stbi_image_free(data);
    return img_data;
}
