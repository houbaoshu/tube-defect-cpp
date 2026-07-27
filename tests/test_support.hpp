#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#define CHECK_TRUE(condition)                                                     \
    do {                                                                          \
        if (!(condition)) {                                                       \
            throw std::runtime_error(                                             \
                std::string("检查失败: ") + #condition + " @ " + __FILE__ + ":" + \
                std::to_string(__LINE__));                                        \
        }                                                                         \
    } while (false)

inline void checkNear(double actual, double expected, double tolerance) {
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream message;
        message << "数值不匹配: actual=" << actual << ", expected=" << expected
                << ", tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}
