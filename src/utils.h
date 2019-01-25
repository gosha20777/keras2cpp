#pragma once
#include <chrono>
#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>

#define stringify(x) #x

#define cast(x) static_cast<ptrdiff_t>(x)

#ifndef NDEBUG
#define kassert_eq(x, y, eps) \
    { \
        auto x_ = static_cast<double>(x); \
        auto y_ = static_cast<double>(y); \
        if (std::abs(x_ - y_) > eps) { \
            printf( \
                "ASSERT [%s:%d] %f isn't equal to %f ('%s' != '%s')\n", \
                __FILE__, __LINE__, x_, y_, stringify(x), stringify(y)); \
            exit(-1); \
        } \
    }
#define kassert(x) \
    if (!(x)) { \
        printf( \
            "ASSERT [%s:%d] '%s' failed\n", __FILE__, __LINE__, stringify(x)); \
        exit(-1); \
    }
#else
#define kassert(x) ;
#define kassert_eq(x, y, eps) ;
#endif

namespace keras2cpp {
    template <typename Callable, typename... Args>
    auto timeit(Callable&& callable, Args&&... args) {
        using namespace std::chrono;
        auto begin = high_resolution_clock::now();
        auto result = [&]() {
            if constexpr (std::is_void_v<std::invoke_result_t<Callable, Args...>>)
                return (std::invoke(callable, args...), nullptr);
            else
                return std::invoke(callable, args...);
        }();
        return std::make_tuple(
            std::move(result),
            duration<double>(high_resolution_clock::now() - begin).count());
    }
}
