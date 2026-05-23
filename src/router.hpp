#include <fstream>
#include <ios>
#include <string>
#include <cstring>
#include <vector>
#include <cctype>

enum FileType {
    TEXT, IMAGE, UNKNOWN
};

enum FileType detect_file_type(const std::string& filepath){

    std::vector<unsigned char> buff(128, 0);
    std::ifstream in_file(filepath, std::ios::binary);

    if(!in_file.is_open()){
        return UNKNOWN;
    }

    in_file.read(reinterpret_cast<char *>(buff.data()), buff.size()*sizeof(char));
    std::streamsize bytes_read = in_file.gcount();

    if(bytes_read == 0){
        return TEXT;
    }

    if(bytes_read >= 3 && buff[0] == 0xff && buff[1] == 0xd8 && buff[2] == 0xff){
        return IMAGE;
    }

    if(bytes_read >= 4 && buff[0] == 0x89 && buff[1] == 0x50 && buff[2] == 0x4e && buff[3] == 0x47){
        return IMAGE;
    }

    for(std::streamsize i=0; i<bytes_read; i++){
        unsigned char ch = buff[i];

        if(std::isprint(ch) || std::isspace(ch)){
            continue;
        }

        return UNKNOWN;
    }

    return TEXT;
}
