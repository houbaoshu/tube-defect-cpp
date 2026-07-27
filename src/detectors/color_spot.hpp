#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace tube_defect::detectors {

struct ColorSpotResult {
    cv::Rect bbox;
    double area{};
    double aspectRatio{};
    double fillRatio{};
    double circularity{};
    std::array<double, 3> meanHsv{};
    double contrast{};
};

[[nodiscard]] cv::Mat makeColorSpotMask(const cv::Mat& image);
[[nodiscard]] std::pair<std::vector<ColorSpotResult>, cv::Mat> detectColorSpots(
    const cv::Mat& image,
    double minArea = 20.0);

}  // namespace tube_defect::detectors
