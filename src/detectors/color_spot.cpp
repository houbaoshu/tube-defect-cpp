#include "detectors/color_spot.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <opencv2/imgproc.hpp>

#include "detail/opencv_geometry.hpp"
#include "detail/statistics.hpp"

namespace tube_defect::detectors {
namespace {

const cv::Scalar kTargetHsvLower{0, 80, 80};
const cv::Scalar kTargetHsvUpper{20, 255, 180};

}  // namespace

cv::Mat makeColorSpotMask(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, kTargetHsvLower, kTargetHsvUpper, mask);
    return mask;
}

std::pair<std::vector<ColorSpotResult>, cv::Mat> detectColorSpots(
    const cv::Mat& image,
    double minArea) {
    cv::Mat hsv;
    cv::Mat gray;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::inRange(hsv, kTargetHsvLower, kTargetHsvUpper, mask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<ColorSpotResult> findings;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < minArea) {
            continue;
        }
        const cv::Rect box = cv::boundingRect(contour);
        const double aspectRatio = static_cast<double>(box.width) / std::max(box.height, 1);
        const double fillRatio = area / std::max(box.area(), 1);
        if (aspectRatio < 0.55 || aspectRatio > 1.80 || fillRatio < 0.25) {
            continue;
        }

        cv::Mat componentMask = cv::Mat::zeros(mask.size(), CV_8U);
        cv::drawContours(componentMask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);
        const double perimeter = cv::arcLength(contour, true);
        const double circularity = perimeter > 0.0
                                       ? 4.0 * std::numbers::pi * area / (perimeter * perimeter)
                                       : 0.0;
        const cv::Scalar meanHsv = cv::mean(hsv, componentMask);

        cv::Mat dilated;
        cv::dilate(
            componentMask,
            dilated,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, {31, 31}));
        cv::Mat ring;
        cv::subtract(dilated, componentMask, ring);
        const auto componentGray = detail::maskedValues(gray, componentMask);
        const auto ringGray = detail::maskedValues(gray, ring);
        const double contrast = componentGray.empty() || ringGray.empty()
                                    ? 0.0
                                    : detail::median(ringGray) - detail::median(componentGray);
        findings.push_back({
            box,
            area,
            aspectRatio,
            fillRatio,
            circularity,
            {meanHsv[0], meanHsv[1], meanHsv[2]},
            contrast,
        });
    }
    std::ranges::sort(findings, std::greater{}, &ColorSpotResult::area);
    return {std::move(findings), mask};
}

}  // namespace tube_defect::detectors
