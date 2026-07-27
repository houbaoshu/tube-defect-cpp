#include "detectors/interface.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "detail/opencv_geometry.hpp"
#include "detail/statistics.hpp"

namespace tube_defect::detectors {
namespace {

const cv::Scalar kBlueLower{100, 120, 25};
const cv::Scalar kBlueUpper{135, 255, 255};

struct ComponentInfo {
    cv::Mat mask;
    cv::Rect bbox;
    int area{};
};

cv::Mat rawBlueMask(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, kBlueLower, kBlueUpper, mask);
    return mask;
}

ComponentInfo largestComponent(const cv::Mat& mask) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    if (count <= 1) {
        throw std::runtime_error("未找到蓝色软管区域");
    }
    int largestLabel = 1;
    for (int label = 2; label < count; ++label) {
        if (stats.at<int>(label, cv::CC_STAT_AREA) >
            stats.at<int>(largestLabel, cv::CC_STAT_AREA)) {
            largestLabel = label;
        }
    }
    cv::Mat component;
    cv::compare(labels, largestLabel, component, cv::CMP_EQ);
    return {
        component,
        {
            stats.at<int>(largestLabel, cv::CC_STAT_LEFT),
            stats.at<int>(largestLabel, cv::CC_STAT_TOP),
            stats.at<int>(largestLabel, cv::CC_STAT_WIDTH),
            stats.at<int>(largestLabel, cv::CC_STAT_HEIGHT),
        },
        stats.at<int>(largestLabel, cv::CC_STAT_AREA),
    };
}

cv::Mat makeBlueTubeMask(const cv::Mat& image) {
    cv::Mat blueMask = rawBlueMask(image);
    int openWidth = std::max(15, static_cast<int>(std::lround(image.cols * 0.07)));
    if (openWidth % 2 == 0) {
        ++openWidth;
    }
    cv::morphologyEx(
        blueMask,
        blueMask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, {openWidth, 5}));
    int closeWidth = std::max(15, static_cast<int>(std::lround(image.cols * 0.04)));
    if (closeWidth % 2 == 0) {
        ++closeWidth;
    }
    cv::morphologyEx(
        blueMask,
        blueMask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {closeWidth, 15}));
    return largestComponent(blueMask).mask;
}

std::vector<int> longestContiguousRun(const std::vector<int>& values) {
    if (values.empty()) {
        return {};
    }
    std::size_t bestBegin = 0;
    std::size_t bestLength = 1;
    std::size_t currentBegin = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] - values[index - 1] > 1) {
            const std::size_t length = index - currentBegin;
            if (length > bestLength) {
                bestBegin = currentBegin;
                bestLength = length;
            }
            currentBegin = index;
        }
    }
    const std::size_t finalLength = values.size() - currentBegin;
    if (finalLength > bestLength) {
        bestBegin = currentBegin;
        bestLength = finalLength;
    }
    return {values.begin() + static_cast<std::ptrdiff_t>(bestBegin),
            values.begin() + static_cast<std::ptrdiff_t>(bestBegin + bestLength)};
}

std::pair<cv::Mat, cv::Mat> extractInterfaceArc(
    const cv::Mat& tubeMask,
    const cv::Mat& edges) {
    std::vector<cv::Point> occupied;
    cv::findNonZero(tubeMask, occupied);
    if (occupied.empty()) {
        throw std::runtime_error("蓝色软管掩膜为空");
    }
    const cv::Rect occupiedBox = cv::boundingRect(occupied);
    const int interfaceTop = occupiedBox.y;
    const int componentWidth = occupiedBox.width;
    const int maxSideDrop = std::max(20, static_cast<int>(std::lround(componentWidth * 0.10)));

    std::vector<int> candidateColumns;
    std::vector<int> topByColumn(tubeMask.cols, -1);
    for (int column = occupiedBox.x; column < occupiedBox.x + occupiedBox.width; ++column) {
        int top = -1;
        for (int row = 0; row < tubeMask.rows; ++row) {
            if (tubeMask.at<std::uint8_t>(row, column) != 0) {
                top = row;
                break;
            }
        }
        if (top >= 0 && top <= interfaceTop + maxSideDrop) {
            candidateColumns.push_back(column);
            topByColumn[column] = top;
        }
    }

    const std::vector<int> columns = longestContiguousRun(candidateColumns);
    const int minimumColumns = std::max(30, static_cast<int>(std::lround(componentWidth * 0.20)));
    if (static_cast<int>(columns.size()) < minimumColumns) {
        throw std::runtime_error("未找到横跨软管的接口弧线");
    }

    std::vector<cv::Point> arc;
    arc.reserve(columns.size());
    for (const int column : columns) {
        const int maskTop = topByColumn[column];
        const int searchTop = std::max(0, maskTop - 5);
        const int searchBottom = std::min(edges.rows, maskTop + 6);
        int selectedRow = maskTop;
        int bestDistance = edges.rows;
        for (int row = searchTop; row < searchBottom; ++row) {
            if (edges.at<std::uint8_t>(row, column) != 0) {
                const int distance = std::abs(row - maskTop);
                if (distance < bestDistance) {
                    selectedRow = row;
                    bestDistance = distance;
                }
            }
        }
        arc.emplace_back(column, selectedRow);
    }

    cv::Mat yValues(static_cast<int>(arc.size()), 1, CV_32F);
    for (int index = 0; index < yValues.rows; ++index) {
        yValues.at<float>(index) = static_cast<float>(arc[index].y);
    }
    cv::medianBlur(yValues, yValues, 5);
    for (int index = 0; index < yValues.rows; ++index) {
        arc[index].y = static_cast<int>(yValues.at<float>(index));
    }

    constexpr int tailWindow = 12;
    const int sidewallThreshold = std::max(
        4,
        static_cast<int>(std::lround(componentWidth * 0.008)));
    while (arc.size() > tailWindow * 2) {
        std::vector<double> values;
        values.reserve(tailWindow);
        for (std::size_t index = arc.size() - tailWindow - 1; index < arc.size() - 1; ++index) {
            values.push_back(arc[index].y);
        }
        if (std::abs(arc.back().y - detail::median(std::move(values))) < sidewallThreshold) {
            break;
        }
        arc.pop_back();
    }
    while (arc.size() > tailWindow * 2) {
        std::vector<double> values;
        values.reserve(tailWindow);
        for (int index = 1; index <= tailWindow; ++index) {
            values.push_back(arc[index].y);
        }
        if (std::abs(arc.front().y - detail::median(std::move(values))) < sidewallThreshold) {
            break;
        }
        arc.erase(arc.begin());
    }

    const cv::Mat contour(arc, true);
    cv::Mat arcMask = cv::Mat::zeros(tubeMask.size(), CV_8U);
    cv::polylines(arcMask, std::vector<std::vector<cv::Point>>{arc}, false, 255, 2, cv::LINE_AA);
    return {contour, arcMask};
}

void shiftArcX(cv::Mat& arc, int offsetX) {
    auto* points = arc.ptr<cv::Point>();
    for (std::size_t index = 0; index < arc.total(); ++index) {
        points[index].x += offsetX;
    }
}

void validateImage(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("输入图像为空");
    }
    if (image.type() != CV_8UC3) {
        throw std::invalid_argument("输入图像必须是 BGR 三通道图像");
    }
}

}  // namespace

bool hasFilledBlueTube(const cv::Mat& image) {
    const cv::Mat mask = rawBlueMask(image);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    if (count <= 1) {
        return false;
    }
    int largestLabel = 1;
    for (int label = 2; label < count; ++label) {
        if (stats.at<int>(label, cv::CC_STAT_AREA) > stats.at<int>(largestLabel, cv::CC_STAT_AREA)) {
            largestLabel = label;
        }
    }
    const int area = stats.at<int>(largestLabel, cv::CC_STAT_AREA);
    const int bboxArea = stats.at<int>(largestLabel, cv::CC_STAT_WIDTH) *
                         stats.at<int>(largestLabel, cv::CC_STAT_HEIGHT);
    const int minimumArea = std::max(
        5'000,
        static_cast<int>(std::lround(image.total() * 0.015)));
    return area >= minimumArea && static_cast<double>(area) / bboxArea >= 0.20;
}

InterfaceCircleResult detectInterfaceCircle(
    const cv::Mat& image,
    double rmseToleranceRatio,
    double p95ToleranceRatio) {
    validateImage(image);

    const ComponentInfo rawComponent = largestComponent(rawBlueMask(image));
    if (image.cols > rawComponent.bbox.width * 2) {
        const int padding = static_cast<int>(std::lround(rawComponent.bbox.width * 0.22));
        const int cropX0 = std::max(0, rawComponent.bbox.x - padding);
        const int cropX1 = std::min(
            image.cols,
            rawComponent.bbox.x + rawComponent.bbox.width + padding);
        auto cropped = detectInterfaceCircle(
            image(cv::Rect{cropX0, 0, cropX1 - cropX0, image.rows}),
            rmseToleranceRatio,
            p95ToleranceRatio);
        shiftArcX(cropped.contour, cropX0);
        shiftArcX(cropped.fittedArc, cropX0);

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Mat edges = cv::Mat::zeros(gray.size(), CV_8U);
        cv::Mat candidateMask = cv::Mat::zeros(gray.size(), CV_8U);
        cropped.edges.copyTo(edges(cv::Rect{cropX0, 0, cropX1 - cropX0, image.rows}));
        cropped.candidateMask.copyTo(
            candidateMask(cv::Rect{cropX0, 0, cropX1 - cropX0, image.rows}));
        cv::Mat visualization = image.clone();
        const cv::Scalar color = cropped.isDistorted ? cv::Scalar{0, 0, 255} : cv::Scalar{0, 220, 0};
        const int thickness = std::max(3, static_cast<int>(std::lround(std::min(image.rows, image.cols) / 600.0)));
        cv::polylines(visualization, cropped.fittedArc, false, color, thickness, cv::LINE_AA);
        cropped.gray = std::move(gray);
        cropped.edges = std::move(edges);
        cropped.candidateMask = std::move(candidateMask);
        cropped.visualization = std::move(visualization);
        return cropped;
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat enhanced;
    cv::createCLAHE(2.0, {8, 8})->apply(gray, enhanced);
    cv::Mat blurred;
    cv::GaussianBlur(enhanced, blurred, {5, 5}, 0.0);
    cv::Mat edges;
    cv::Canny(blurred, edges, 30, 90);
    const cv::Mat tubeMask = makeBlueTubeMask(image);
    auto [contour, candidateMask] = extractInterfaceArc(tubeMask, edges);

    std::vector<cv::Point2d> points;
    points.reserve(contour.total());
    const auto* contourPoints = contour.ptr<cv::Point>();
    for (std::size_t index = 0; index < contour.total(); ++index) {
        points.emplace_back(contourPoints[index].x, contourPoints[index].y);
    }
    const detail::QuadraticFit fit = detail::robustQuadraticFit(points);
    std::vector<cv::Point> fittedPoints;
    fittedPoints.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        fittedPoints.emplace_back(
            static_cast<int>(std::nearbyint(points[index].x)),
            static_cast<int>(std::nearbyint(fit.fittedY[index])));
    }
    cv::Mat fittedArc(fittedPoints, true);

    std::vector<double> deviations(fit.residuals.size());
    std::transform(
        fit.residuals.begin(), fit.residuals.end(), deviations.begin(),
        [](double value) { return std::abs(value); });
    const double sumSquares = std::inner_product(
        fit.residuals.begin(), fit.residuals.end(), fit.residuals.begin(), 0.0);
    const double rmse = std::sqrt(sumSquares / fit.residuals.size());
    const double p95Deviation = detail::percentile(deviations, 95.0);
    const double maxDeviation = *std::max_element(deviations.begin(), deviations.end());
    const double arcWidth = points.back().x - points.front().x + 1.0;
    const double rmseTolerance = std::max(2.0, arcWidth * rmseToleranceRatio);
    const double p95Tolerance = std::max(4.0, arcWidth * p95ToleranceRatio);
    const bool isDistorted = rmse > rmseTolerance && p95Deviation > p95Tolerance;

    cv::Mat visualization = image.clone();
    const cv::Scalar color = isDistorted ? cv::Scalar{0, 0, 255} : cv::Scalar{0, 220, 0};
    const int thickness = std::max(3, static_cast<int>(std::lround(std::min(image.rows, image.cols) / 600.0)));
    cv::polylines(visualization, fittedArc, false, color, thickness, cv::LINE_AA);

    return {
        contour,
        fittedArc,
        fit.coefficients,
        rmse,
        p95Deviation,
        maxDeviation,
        rmseTolerance,
        p95Tolerance,
        isDistorted,
        gray,
        edges,
        candidateMask,
        visualization,
    };
}

}  // namespace tube_defect::detectors
