#include "visualization.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/imgproc.hpp>

namespace tube_defect {
namespace {

cv::Scalar colorFor(DefectType type) {
    switch (type) {
        case DefectType::interfaceDistortion:
            return {0, 0, 255};
        case DefectType::colorSpot:
            return {0, 190, 255};
        case DefectType::scratch:
            return {0, 255, 0};
        case DefectType::hole:
        default:
            return {255, 80, 0};
    }
}

std::string_view labelFor(DefectType type) {
    switch (type) {
        case DefectType::interfaceDistortion:
            return "INTERFACE";
        case DefectType::colorSpot:
            return "COLOR SPOT";
        case DefectType::scratch:
            return "SCRATCH";
        case DefectType::hole:
        default:
            return "HOLE";
    }
}

}  // namespace

cv::Mat annotate(
    const cv::Mat& image,
    const std::vector<DefectFinding>& findings,
    DetectionStatus status) {
    cv::Mat result = image.clone();
    const int reference = std::min(image.rows, image.cols);
    const int thickness = std::max(2, static_cast<int>(std::lround(reference / 700.0)));
    const double fontScale = std::max(0.55, reference / 1800.0);
    for (std::size_t index = 0; index < findings.size(); ++index) {
        const auto& finding = findings[index];
        const cv::Scalar color = colorFor(finding.type);
        cv::rectangle(
            result,
            {finding.bbox.x, finding.bbox.y},
            {finding.bbox.right(), finding.bbox.bottom()},
            color,
            thickness);
        const cv::Point center = finding.bbox.center();
        cv::drawMarker(
            result,
            center,
            color,
            cv::MARKER_CROSS,
            std::max(14, thickness * 6),
            thickness);
        const std::string label = std::string(labelFor(finding.type)) + " #" +
                                  std::to_string(index + 1) + " (" +
                                  std::to_string(center.x) + "," +
                                  std::to_string(center.y) + ")";
        cv::putText(
            result,
            label,
            {finding.bbox.x, std::max(30, finding.bbox.y - 10)},
            cv::FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            thickness,
            cv::LINE_AA);
    }

    const bool isDefect = status == DetectionStatus::defect;
    cv::putText(
        result,
        std::string("RESULT: ") + (isDefect ? "DEFECT" : "NORMAL"),
        {24, std::max(45, static_cast<int>(std::lround(reference * 0.035)))},
        cv::FONT_HERSHEY_SIMPLEX,
        std::max(0.8, fontScale * 1.15),
        isDefect ? cv::Scalar{0, 0, 200} : cv::Scalar{0, 150, 0},
        std::max(2, thickness),
        cv::LINE_AA);
    return result;
}

}  // namespace tube_defect
