#include "detectors/scratch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <opencv2/imgproc.hpp>

namespace tube_defect::detectors {
namespace {

const cv::Scalar kMarkerLower{0, 90, 80};
const cv::Scalar kMarkerUpper{20, 255, 255};

struct RoiInfo {
    cv::Mat image;
    cv::Point offset;
};

RoiInfo markerGuidedRoi(const cv::Mat& image, const OrangeMarker& marker) {
    const double centerY = marker.bbox.y + marker.bbox.height / 2.0;
    const int y0 = std::max(0, static_cast<int>(std::lround(centerY - marker.bbox.height * 0.63)));
    const int y1 = std::min(image.rows, static_cast<int>(std::lround(centerY + marker.bbox.height * 0.54)));
    const int x0 = std::max(0, marker.bbox.x - marker.bbox.height);
    return {image(cv::Rect{x0, y0, image.cols - x0, y1 - y0}), {x0, y0}};
}

cv::Mat directionMask(
    const cv::Mat& gradientX,
    const cv::Mat& gradientY,
    double minAngle,
    double maxAngle) {
    cv::Mat result = cv::Mat::zeros(gradientX.size(), CV_8U);
    for (int y = 0; y < gradientX.rows; ++y) {
        const auto* xRow = gradientX.ptr<float>(y);
        const auto* yRow = gradientY.ptr<float>(y);
        auto* output = result.ptr<std::uint8_t>(y);
        for (int x = 0; x < gradientX.cols; ++x) {
            double gradientAngle = std::atan2(yRow[x], xRow[x]) * 180.0 / std::numbers::pi;
            gradientAngle = std::fmod(gradientAngle + 180.0, 180.0);
            const double edgeAngle = std::fmod(gradientAngle + 90.0, 180.0);
            const double acuteAngle = std::min(edgeAngle, 180.0 - edgeAngle);
            if (acuteAngle >= minAngle && acuteAngle <= maxAngle) {
                output[x] = 255;
            }
        }
    }
    return result;
}

cv::Mat normalizeToUint8Truncate(const cv::Mat& magnitude) {
    cv::Mat normalized;
    cv::normalize(magnitude, normalized, 0, 255, cv::NORM_MINMAX);
    cv::Mat result(normalized.size(), CV_8U);
    for (int y = 0; y < normalized.rows; ++y) {
        const auto* input = normalized.ptr<float>(y);
        auto* output = result.ptr<std::uint8_t>(y);
        for (int x = 0; x < normalized.cols; ++x) {
            output[x] = static_cast<std::uint8_t>(std::clamp(input[x], 0.0F, 255.0F));
        }
    }
    return result;
}

ScratchResult lineFeatures(const cv::Mat& componentMask, cv::Rect bbox, int area) {
    std::vector<cv::Point> nonZero;
    cv::findNonZero(componentMask, nonZero);
    cv::Mat points(static_cast<int>(nonZero.size()), 2, CV_32F);
    for (int index = 0; index < points.rows; ++index) {
        points.at<float>(index, 0) = static_cast<float>(nonZero[index].x);
        points.at<float>(index, 1) = static_cast<float>(nonZero[index].y);
    }
    const cv::PCA pca(points, cv::Mat{}, cv::PCA::DATA_AS_ROW);
    const cv::Vec2f major{
        pca.eigenvectors.at<float>(0, 0),
        pca.eigenvectors.at<float>(0, 1),
    };
    const cv::Vec2f minor{
        pca.eigenvectors.at<float>(1, 0),
        pca.eigenvectors.at<float>(1, 1),
    };
    float majorMin = std::numeric_limits<float>::max();
    float majorMax = std::numeric_limits<float>::lowest();
    float minorMin = std::numeric_limits<float>::max();
    float minorMax = std::numeric_limits<float>::lowest();
    const cv::Vec2f mean{pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1)};
    for (const cv::Point point : nonZero) {
        const cv::Vec2f centered{
            static_cast<float>(point.x) - mean[0],
            static_cast<float>(point.y) - mean[1],
        };
        const float majorProjection = centered.dot(major);
        const float minorProjection = centered.dot(minor);
        majorMin = std::min(majorMin, majorProjection);
        majorMax = std::max(majorMax, majorProjection);
        minorMin = std::min(minorMin, minorProjection);
        minorMax = std::max(minorMax, minorProjection);
    }
    const double length = majorMax - majorMin + 1.0;
    const double width = minorMax - minorMin + 1.0;
    double angle = std::atan2(major[1], major[0]) * 180.0 / std::numbers::pi;
    angle = std::fmod(angle + 180.0, 180.0);
    const double acuteAngle = std::min(angle, 180.0 - angle);
    return {bbox, area, length, width, length / std::max(width, 1.0), acuteAngle};
}

}  // namespace

std::optional<OrangeMarker> findLargeOrangeMarker(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, kMarkerLower, kMarkerUpper, mask);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    if (count <= 1) {
        return std::nullopt;
    }
    int largestLabel = 1;
    for (int label = 2; label < count; ++label) {
        if (stats.at<int>(label, cv::CC_STAT_AREA) > stats.at<int>(largestLabel, cv::CC_STAT_AREA)) {
            largestLabel = label;
        }
    }
    const int area = stats.at<int>(largestLabel, cv::CC_STAT_AREA);
    const int minimumArea = std::max(
        800,
        static_cast<int>(std::lround(image.total() * 0.0008)));
    if (area < minimumArea) {
        return std::nullopt;
    }
    cv::Mat component;
    cv::compare(labels, largestLabel, component, cv::CMP_EQ);
    return OrangeMarker{
        {
            stats.at<int>(largestLabel, cv::CC_STAT_LEFT),
            stats.at<int>(largestLabel, cv::CC_STAT_TOP),
            stats.at<int>(largestLabel, cv::CC_STAT_WIDTH),
            stats.at<int>(largestLabel, cv::CC_STAT_HEIGHT),
        },
        area,
        component,
    };
}

ScratchDetectionResult detectScratches(
    const cv::Mat& image,
    const std::optional<OrangeMarker>& suppliedMarker,
    double minAngle,
    double maxAngle) {
    const auto marker = suppliedMarker ? suppliedMarker : findLargeOrangeMarker(image);
    if (!marker) {
        return {{}, cv::Mat::zeros(image.size(), CV_8U), image};
    }
    const RoiInfo roiInfo = markerGuidedRoi(image, *marker);
    const cv::Mat& roi = roiInfo.image;
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, {5, 5}, 0.0);
    cv::Mat canny;
    cv::Canny(blurred, canny, 30, 80);

    cv::Mat sobelX;
    cv::Mat sobelY;
    cv::Sobel(blurred, sobelX, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, sobelY, CV_32F, 0, 1, 3);
    cv::Mat scharrX;
    cv::Mat scharrY;
    cv::Scharr(blurred, scharrX, CV_32F, 1, 0);
    cv::Scharr(blurred, scharrY, CV_32F, 0, 1);

    cv::Mat sobelMagnitude;
    cv::magnitude(sobelX, sobelY, sobelMagnitude);
    const cv::Mat sobel = normalizeToUint8Truncate(sobelMagnitude);
    cv::Mat scharrMagnitude;
    cv::magnitude(scharrX, scharrY, scharrMagnitude);
    const cv::Mat scharr = normalizeToUint8Truncate(scharrMagnitude);

    const cv::Mat sobelDirection = directionMask(sobelX, sobelY, minAngle, maxAngle);
    const cv::Mat scharrDirection = directionMask(scharrX, scharrY, minAngle, maxAngle);
    cv::Mat validSurface;
    cv::compare(gray, 220, validSurface, cv::CMP_LE);
    cv::erode(
        validSurface,
        validSurface,
        cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    cv::Mat innerTube = cv::Mat::zeros(gray.size(), CV_8U);
    const int marginY = static_cast<int>(std::lround(gray.rows * 0.18));
    const int marginX = static_cast<int>(std::lround(gray.cols * 0.10));
    if (gray.rows > marginY * 2 && gray.cols > marginX * 2) {
        innerTube(cv::Rect{
            marginX,
            marginY,
            gray.cols - marginX * 2,
            gray.rows - marginY * 2}) = 255;
    }
    cv::bitwise_and(validSurface, innerTube, validSurface);

    cv::Mat cannyDiagonal;
    cv::Mat sobelDiagonal;
    cv::Mat scharrDiagonal;
    cv::bitwise_and(canny, sobelDirection, cannyDiagonal);
    cv::bitwise_and(sobel, sobelDirection, sobelDiagonal);
    cv::bitwise_and(scharr, scharrDirection, scharrDiagonal);
    cv::bitwise_and(cannyDiagonal, validSurface, cannyDiagonal);
    cv::bitwise_and(sobelDiagonal, validSurface, sobelDiagonal);
    cv::bitwise_and(scharrDiagonal, validSurface, scharrDiagonal);
    cv::Mat sobelBinary;
    cv::Mat scharrBinary;
    cv::threshold(sobelDiagonal, sobelBinary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(scharrDiagonal, scharrBinary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat combined;
    cv::bitwise_or(cannyDiagonal, sobelBinary, combined);
    cv::bitwise_or(combined, scharrBinary, combined);
    cv::morphologyEx(
        combined,
        combined,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(combined, labels, stats, centroids);
    const int minimumArea = std::max(
        80,
        static_cast<int>(std::lround(roi.total() * 0.0010)));
    std::vector<ScratchResult> findings;
    cv::Mat fullMask = cv::Mat::zeros(image.size(), CV_8U);
    const int minimumOriginalX = marker->bbox.x + marker->bbox.width +
                                 static_cast<int>(std::lround(marker->bbox.height * 0.5));
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < minimumArea) {
            continue;
        }
        cv::Mat component;
        cv::compare(labels, label, component, cv::CMP_EQ);
        const cv::Rect localBox{
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT),
        };
        const cv::Rect originalBox{
            localBox.x + roiInfo.offset.x,
            localBox.y + roiInfo.offset.y,
            localBox.width,
            localBox.height,
        };
        ScratchResult features = lineFeatures(component, originalBox, area);
        if (features.aspectRatio < 1.20 || originalBox.x < minimumOriginalX) {
            continue;
        }
        fullMask(cv::Rect{roiInfo.offset.x, roiInfo.offset.y, roi.cols, roi.rows})
            .setTo(255, component);
        findings.push_back(features);
    }
    std::ranges::sort(findings, std::greater{}, &ScratchResult::area);
    return {std::move(findings), std::move(fullMask), roi};
}

}  // namespace tube_defect::detectors
