#pragma once

#include <filesystem>
#include <string>

#include <opencv2/core/mat.hpp>

#include "tube_defect/models.hpp"

namespace tube_defect {

[[nodiscard]] DetectionResult analyzeImage(
    const cv::Mat& image,
    std::string imagePath = "<memory>");
[[nodiscard]] DetectionResult analyzePath(const std::filesystem::path& imagePath);

}  // namespace tube_defect
