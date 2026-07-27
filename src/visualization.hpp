#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "tube_defect/models.hpp"

namespace tube_defect {

[[nodiscard]] cv::Mat annotate(
    const cv::Mat& image,
    const std::vector<DefectFinding>& findings,
    DetectionStatus status);

}  // namespace tube_defect
