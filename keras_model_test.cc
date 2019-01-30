#include "test/benchmark.h"
#include "test/conv_2x2.h"
#include "test/conv_3x3.h"
#include "test/conv_3x3x3.h"
#include "test/conv_hard_sigmoid_2x2.h"
#include "test/conv_sigmoid_2x2.h"
#include "test/conv_softplus_2x2.h"
#include "test/dense_10x1.h"
#include "test/dense_10x10.h"
#include "test/dense_10x10x10.h"
#include "test/dense_1x1.h"
#include "test/dense_2x2.h"
#include "test/dense_relu_10.h"
#include "test/dense_tanh_10.h"
#include "test/elu_10.h"
#include "test/embedding_64.h"
#include "test/lstm_simple_7x20.h"
#include "test/lstm_simple_stacked_16x9.h"
#include "test/lstm_stacked_64x83.h"
#include "test/maxpool2d_1x1.h"
#include "test/maxpool2d_2x2.h"
#include "test/maxpool2d_3x2x2.h"
#include "test/maxpool2d_3x3x3.h"
#include "test/relu_10.h"
#include "src/model.h"

using namespace keras2cpp;

namespace test {
    inline void basics() noexcept {
        {
            const int i = 3;
            const int j = 5;
            const int k = 10;
            Tensor t {i, j, k};

            float c = 1.f;
            for (size_t ii = 0; ii < i; ++ii)
                for (size_t jj = 0; jj < j; ++jj)
                    for (size_t kk = 0; kk < k; ++kk) {
                        t(ii, jj, kk) = c;
                        c += 1.f;
                    }
            c = 1.f;
            size_t cc = 0;
            for (size_t ii = 0; ii < i; ++ii)
                for (size_t jj = 0; jj < j; ++jj)
                    for (size_t kk = 0; kk < k; ++kk) {
                        kassert_eq(t(ii, jj, kk), c, 1e-9);
                        kassert_eq(t.data_[cc], c, 1e-9);
                        c += 1.f;
                        ++cc;
                    }
        }
        {
            const size_t i = 2;
            const size_t j = 3;
            const size_t k = 4;
            const size_t l = 5;
            Tensor t {i, j, k, l};

            float c = 1.f;
            for (size_t ii = 0; ii < i; ++ii)
                for (size_t jj = 0; jj < j; ++jj)
                    for (size_t kk = 0; kk < k; ++kk)
                        for (size_t ll = 0; ll < l; ++ll) {
                            t(ii, jj, kk, ll) = c;
                            c += 1.f;
                        }
            c = 1.f;
            size_t cc = 0;
            for (size_t ii = 0; ii < i; ++ii)
                for (size_t jj = 0; jj < j; ++jj)
                    for (size_t kk = 0; kk < k; ++kk)
                        for (size_t ll = 0; ll < l; ++ll) {
                            kassert_eq(t(ii, jj, kk, ll), c, 1e-9);
                            kassert_eq(t.data_[cc], c, 1e-9);
                            c += 1.f;
                            ++cc;
                        }
        }
        {
            Tensor a {2, 2};
            Tensor b {2, 2};

            a.data_ = {1.0, 2.0, 3.0, 5.0};
            b.data_ = {2.0, 5.0, 4.0, 1.0};

            Tensor result = a + b;
            kassert(result.data_ == std::vector<float>({3.0, 7.0, 7.0, 6.0}));
        }
        {
            Tensor a {2, 2};
            Tensor b {2, 2};

            a.data_ = {1.0, 2.0, 3.0, 5.0};
            b.data_ = {2.0, 5.0, 4.0, 1.0};

            Tensor result = a * b;
            kassert(result.data_ == std::vector<float>({2.0, 10.0, 12.0, 5.0}));
        }
        {
            Tensor a {1, 2};
            Tensor b {1, 2};

            a.data_ = {1.0, 2.0};
            b.data_ = {2.0, 5.0};

            Tensor result = a.dot(b);
            kassert(result.data_ == std::vector<float>({12.0}));
        }
        {
            Tensor a {2, 1};
            Tensor b {2, 1};

            a.data_ = {1.0, 2.0};
            b.data_ = {2.0, 5.0};

            Tensor result = a.dot(b);
            kassert(result.data_ == std::vector<float>({2.0, 5.0, 4.0, 10.0}));
        }
    }
}

int main() {
    test::basics();
    test::dense_1x1();
    test::dense_10x1();
    test::dense_2x2();
    test::dense_10x10();
    test::dense_10x10x10();
    test::conv_2x2();
    test::conv_3x3();
    test::conv_3x3x3();
    test::elu_10();
    test::relu_10();
    test::dense_relu_10();
    test::dense_tanh_10();
    test::conv_softplus_2x2();
    test::conv_hard_sigmoid_2x2();
    test::conv_sigmoid_2x2();
    test::maxpool2d_1x1();
    test::maxpool2d_2x2();
    test::maxpool2d_3x2x2();
    test::maxpool2d_3x3x3();
    test::lstm_simple_7x20();
    test::lstm_simple_stacked_16x9();
    test::lstm_stacked_64x83();
    test::embedding_64();

    const size_t n = 10; // Run benchmark "n" times.

    double total_load_time = 0.0;
    double total_apply_time = 0.0;

    for (size_t i = 0; i < n; ++i) {
        auto [load_time, apply_time] = test::benchmark();
        total_load_time += load_time;
        total_apply_time += apply_time;
    }
    printf("Benchmark network loads in %fs\n", total_load_time / n);
    printf("Benchmark network runs in %fs\n", total_apply_time / n);

    return 0;
}