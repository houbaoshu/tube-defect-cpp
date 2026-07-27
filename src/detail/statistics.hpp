#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace tube_defect::detail {

struct QuadraticFit {
    std::array<double, 3> coefficients{};
    std::vector<double> fittedY;
    std::vector<double> residuals;
};

[[nodiscard]] double median(std::vector<double> values);
[[nodiscard]] double percentile(std::vector<double> values, double q);
[[nodiscard]] std::vector<double> maskedValues(
    const cv::Mat& values,
    const cv::Mat& mask);
[[nodiscard]] QuadraticFit robustQuadraticFit(
    const std::vector<cv::Point2d>& points,
    int iterations = 6);

}  // namespace tube_defect::detail
