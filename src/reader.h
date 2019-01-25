#pragma once
#include <memory>
#include <type_traits>

namespace keras2cpp {
    class Stream {
        class _Impl;
        std::unique_ptr<_Impl> impl_;
    public:
        Stream(const std::string&);
        ~Stream();

        Stream& reads(char*, size_t);
        template <
            typename T,
            typename = std::enable_if_t<std::is_default_constructible_v<T>>>
        operator T() noexcept {
            T value;
            reads(reinterpret_cast<char*>(&value), sizeof(T));
            return value;
        }
    };
}