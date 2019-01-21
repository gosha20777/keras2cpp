#pragma once
#ifndef KERAS_MODEL_H_
#define KERAS_MODEL_H_

#include <algorithm>
#include <chrono>
#include <math.h>
#include <numeric>
#include <string>
#include <vector>

#define KASSERT(x, ...)                                                        \
    if (!(x)) {                                                                \
        printf("KASSERT: %s(%d): ", __FILE__, __LINE__);                       \
        printf(__VA_ARGS__);                                                   \
        printf("\n");                                                          \
        return false;                                                          \
    }

#define KASSERT_EQ(x, y, eps)                                                  \
    if (fabs(x - y) > eps) {                                                   \
        printf("KASSERT: Expected %f, got %f\n", y, x);                        \
        return false;                                                          \
    }

#ifdef DEBUG
#define KDEBUG(x, ...)                                                         \
    if (!(x)) {                                                                \
        printf("%s(%d): ", __FILE__, __LINE__);                                \
        printf(__VA_ARGS__);                                                   \
        printf("\n");                                                          \
        exit(-1);                                                              \
    }
#else
#define KDEBUG(x, ...) ;
#endif
#endif

namespace keras2cpp {
    class KerasTimer {
        public:
            KerasTimer() {}
            void Start() { start_ = std::chrono::high_resolution_clock::now(); }

            double Stop() {
                std::chrono::time_point<std::chrono::high_resolution_clock> now =
                                        std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = now - start_;
                return diff.count();
            }
        private:
            std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };
}