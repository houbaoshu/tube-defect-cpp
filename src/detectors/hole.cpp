#include "detectors/hole.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "detail/opencv_geometry.hpp"
#include "detail/statistics.hpp"

namespace tube_defect::detectors {
namespace {

struct AxisFit {
    double slope{};
    double intercept{};
    std::vector<double> deviations;
    std::size_t pointCount{};
};

AxisFit fitBlueAxis(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const int lowerY = static_cast<int>(std::lround(image.rows * 0.04));
    const int upperY = static_cast<int>(std::lround(image.rows * 0.75));
    std::vector<cv::Point2d> points;
    for (int y = 0; y < image.rows; ++y) {
        const auto* imageRow = image.ptr<cv::Vec3b>(y);
        const auto* grayRow = gray.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            const int blue = imageRow[x][0];
            const int red = imageRow[x][2];
            if (blue - red > 20 && blue > 90 && grayRow[x] > 70 && y > lowerY && y < upperY) {
                points.emplace_back(y, x);
            }
        }
    }
    if (points.size() < 100) {
        return {0.0, image.cols / 2.0, {}, points.size()};
    }

    double sumY = 0.0;
    double sumX = 0.0;
    double sumYY = 0.0;
    double sumYX = 0.0;
    for (const auto& point : points) {
        sumY += point.x;
        sumX += point.y;
        sumYY += point.x * point.x;
        sumYX += point.x * point.y;
    }
    const double count = static_cast<double>(points.size());
    const double denominator = count * sumYY - sumY * sumY;
    const double slope = denominator == 0.0
                             ? 0.0
                             : (count * sumYX - sumY * sumX) / denominator;
    const double intercept = (sumX - slope * sumY) / count;
    std::vector<double> deviations;
    deviations.reserve(points.size());
    for (const auto& point : points) {
        deviations.push_back(std::abs(point.y - (slope * point.x + intercept)));
    }
    return {slope, intercept, std::move(deviations), points.size()};
}

std::pair<cv::Mat, int> blueAxisCrop(const cv::Mat& image) {
    const AxisFit fit = fitBlueAxis(image);
    if (fit.pointCount < 100) {
        return {image, 0};
    }
    const int margin = std::max(
        40,
        static_cast<int>(std::lround(detail::percentile(fit.deviations, 90.0) * 0.55)));
    const double first = fit.intercept;
    const double last = fit.slope * (image.rows - 1) + fit.intercept;
    const int x0 = std::max(0, static_cast<int>(std::lround(std::min(first, last) - margin)));
    const int x1 = std::min(image.cols, static_cast<int>(std::lround(std::max(first, last) + margin)));
    if (x1 - x0 < 80) {
        return {image, 0};
    }
    return {image(cv::Rect{x0, 0, x1 - x0, image.rows}), x0};
}

cv::Mat makeDetectionMask(const cv::Mat& image) {
    const AxisFit fit = fitBlueAxis(image);
    const double slope = fit.pointCount >= 100 ? fit.slope : 0.0;
    const double intercept = fit.pointCount >= 100 ? fit.intercept : image.cols / 2.0;
    const int halfWidth = std::max(40, static_cast<int>(std::lround(image.cols * 0.15)));
    const int lowerY = static_cast<int>(std::lround(image.rows * 0.08));
    const int upperY = static_cast<int>(std::lround(image.rows * 0.75));
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    for (int y = lowerY; y <= upperY && y < image.rows; ++y) {
        if (y < 0) {
            continue;
        }
        const double axisX = slope * y + intercept;
        auto* row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (std::abs(x - axisX) <= halfWidth) {
                row[x] = 255;
            }
        }
    }
    return mask;
}

}  // namespace

HoleDetectionResult detectHoles(const cv::Mat& image) {
    auto [workingImage, offsetX] = blueAxisCrop(image);
    cv::Mat gray;
    cv::cvtColor(workingImage, gray, cv::COLOR_BGR2GRAY);
    const cv::Mat detectionMask = makeDetectionMask(workingImage);

    cv::Mat denoised8;
    cv::GaussianBlur(gray, denoised8, {5, 5}, 0.0);
    cv::Mat denoised;
    denoised8.convertTo(denoised, CV_32F);
    cv::Mat background;
    cv::GaussianBlur(denoised, background, {0, 0}, 16.0, 16.0);
    cv::Mat residualResponse;
    cv::subtract(background, denoised, residualResponse);
    cv::max(residualResponse, 0.0, residualResponse);
    const double residualThreshold = std::max(
        detail::percentile(detail::maskedValues(residualResponse, detectionMask), 98.8),
        20.0);
    cv::Mat residualMask;
    cv::compare(residualResponse, residualThreshold, residualMask, cv::CMP_GE);

    cv::Mat source;
    gray.convertTo(source, CV_32F, 1.0 / 255.0);
    cv::Mat logResponse = cv::Mat::zeros(gray.size(), CV_32F);
    for (const double sigma : {1.5, 2.5, 4.0, 6.0}) {
        cv::Mat blurred;
        cv::GaussianBlur(source, blurred, {0, 0}, sigma, sigma);
        cv::Mat laplacian;
        cv::Laplacian(blurred, laplacian, CV_32F, 3);
        cv::max(laplacian, 0.0, laplacian);
        laplacian *= sigma * sigma;
        cv::max(logResponse, laplacian, logResponse);
    }
    const double logThreshold = detail::percentile(
        detail::maskedValues(logResponse, detectionMask),
        99.0);
    cv::Mat logMask;
    cv::compare(logResponse, logThreshold, logMask, cv::CMP_GE);

    cv::Mat holeMask;
    cv::bitwise_and(residualMask, logMask, holeMask);
    cv::bitwise_and(holeMask, detectionMask, holeMask);
    cv::morphologyEx(
        holeMask,
        holeMask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::morphologyEx(
        holeMask,
        holeMask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(holeMask, labels, stats, centroids);
    std::vector<HoleResult> findings;
    cv::Mat fullMask = cv::Mat::zeros(image.size(), CV_8U);
    for (int label = 1; label < count; ++label) {
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < 40 || area > 1500 || std::min(width, height) < 3 ||
            std::max(width, height) > 25 ||
            static_cast<double>(std::max(width, height)) / std::min(width, height) > 4.0) {
            continue;
        }
        const double fillRatio = static_cast<double>(area) / (width * height);
        if (fillRatio < 0.12) {
            continue;
        }

        cv::Mat component;
        cv::compare(labels, label, component, cv::CMP_EQ);
        const double medianGray = detail::median(detail::maskedValues(gray, component));
        if (medianGray > 175.0) {
            continue;
        }
        cv::Mat dilated;
        cv::dilate(
            component,
            dilated,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, {17, 17}));
        cv::Mat ring;
        cv::subtract(dilated, component, ring);
        const double ringMedian = detail::median(detail::maskedValues(gray, ring));
        const double contrast = ringMedian - medianGray;
        if (contrast < 20.0) {
            continue;
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        const auto largestContour = std::ranges::max_element(
            contours,
            {},
            [](const std::vector<cv::Point>& contour) { return cv::contourArea(contour); });
        const double perimeter = cv::arcLength(*largestContour, true);
        const double contourArea = cv::contourArea(*largestContour);
        const double circularity = perimeter > 0.0
                                       ? 4.0 * std::numbers::pi * contourArea / (perimeter * perimeter)
                                       : 0.0;
        fullMask(cv::Rect{offsetX, 0, workingImage.cols, workingImage.rows})
            .setTo(255, component);
        findings.push_back({
            {x + offsetX, y, width, height},
            area,
            fillRatio,
            circularity,
            medianGray,
            ringMedian,
            contrast,
            std::sqrt(4.0 * area / std::numbers::pi),
        });
    }
    std::ranges::sort(findings, std::greater{}, &HoleResult::area);
    return {std::move(findings), std::move(fullMask), workingImage};
}

}  // namespace tube_defect::detectors
