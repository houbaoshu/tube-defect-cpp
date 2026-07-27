#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace tube_defect::detectors {

struct OrangeMarker {
    cv::Rect bbox;
    int area{};
    cv::Mat mask;
};

struct ScratchResult {
    cv::Rect bbox;
    int area{};
    double length{};
    double width{};
    double aspectRatio{};
    double angleDegrees{};
};

struct ScratchDetectionResult {
    std::vector<ScratchResult> findings;
    cv::Mat mask;
    cv::Mat roi;
};

[[nodiscard]] std::optional<OrangeMarker> findLargeOrangeMarker(const cv::Mat& image);
[[nodiscard]] ScratchDetectionResult detectScratches(
    const cv::Mat& image,
    const std::optional<OrangeMarker>& marker = std::nullopt,
    double minAngle = 15.0,
    double maxAngle = 75.0);

}  // namespace tube_defect::detectors
