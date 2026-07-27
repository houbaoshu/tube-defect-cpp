#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace tube_defect::detectors {

struct HoleResult {
    cv::Rect bbox;
    int area{};
    double fillRatio{};
    double circularity{};
    double medianGray{};
    double backgroundMedianGray{};
    double contrast{};
    double equivalentDiameter{};
};

struct HoleDetectionResult {
    std::vector<HoleResult> findings;
    cv::Mat mask;
    cv::Mat workingImage;
};

[[nodiscard]] HoleDetectionResult detectHoles(const cv::Mat& image);

}  // namespace tube_defect::detectors
