#include "utils.h"
bool ReadUnsignedInt(std::ifstream* file, unsigned int* i) {
    KASSERT(file, "Invalid file stream");
    KASSERT(i, "Invalid pointer");

    file->read((char*)i, sizeof(unsigned int));
    KASSERT(file->gcount() == sizeof(unsigned int), "Expected unsigned int");

    return true;
}

bool ReadFloat(std::ifstream* file, float* f) {
    KASSERT(file, "Invalid file stream");
    KASSERT(f, "Invalid pointer");

    file->read((char*)f, sizeof(float));
    KASSERT(file->gcount() == sizeof(float), "Expected float");

    return true;
}

bool ReadFloats(std::ifstream* file, float* f, size_t n) {
    KASSERT(file, "Invalid file stream");
    KASSERT(f, "Invalid pointer");

    file->read((char*)f, sizeof(float) * n);
    KASSERT(((unsigned int)file->gcount()) == sizeof(float) * n,
            "Expected floats");

    return true;
}