#pragma once

#include <array>

#include <opencv2/core.hpp>

namespace tube_defect::detectors {

struct InterfaceCircleResult {
    cv::Mat contour;
    cv::Mat fittedArc;
    std::array<double, 3> arcCoefficients{};
    double rmse{};
    double p95Deviation{};
    double maxDeviation{};
    double rmseTolerance{};
    double p95Tolerance{};
    bool isDistorted{};
    cv::Mat gray;
    cv::Mat edges;
    cv::Mat candidateMask;
    cv::Mat visualization;
};

[[nodiscard]] bool hasFilledBlueTube(const cv::Mat& image);
[[nodiscard]] InterfaceCircleResult detectInterfaceCircle(
    const cv::Mat& image,
    double rmseToleranceRatio = 0.014,
    double p95ToleranceRatio = 0.030);

}  // namespace tube_defect::detectors
