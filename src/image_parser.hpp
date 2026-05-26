#include "stb_image.h"
#include "stb_image_resize2.h"

#include <string>
#include <vector>

std::vector<unsigned char> process_image(const std::string& filepath, int &width, int &height){
    int channels;
    unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &channels, 3);
    if(data == nullptr){
        return {};
    }

    std::vector<unsigned char> img_data(data, data + width * height * 3);

    stbi_image_free(data);
    return img_data;
}

std::vector<unsigned char> resize_image(unsigned char *img, int w, int h){

    std::vector<unsigned char> resized_img(224 * 224 * 3);
    stbir_resize_uint8_linear(img, w, h, 0, resized_img.data(), 224, 224, 0, STBIR_RGB);

    return resized_img;
}

std::vector<float> normalize_image(const std::vector<unsigned char>& resized_img){
    std::vector<float> tensor_data(224*224*3);

    // Standard CLIP Normalization Constants
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float std_dev[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    for(int y = 0; y < 224; y++){
        for(int x = 0; x < 224; x++){
            for(int c = 0; c < 3; c++){

                int hwc_idx = (y * 224 + x) * 3 + c;

                int nchw_idx = c * (224*224) + (y * 224 + x);

                float pixel = static_cast<float>(resized_img[hwc_idx]);

                tensor_data[nchw_idx] = (pixel / 255.0f - mean[c]) / std_dev[c];
            }
        }
    }

    return tensor_data;
}
