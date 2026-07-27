#include "detail/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tube_defect::detail {
namespace {

std::array<double, 3> solveQuadratic(
    const std::vector<double>& normalizedX,
    const std::vector<double>& y,
    const std::vector<double>& weights) {
    const int count = static_cast<int>(normalizedX.size());
    cv::Mat design(count, 3, CV_64F);
    cv::Mat observations(count, 1, CV_64F);
    for (int index = 0; index < count; ++index) {
        const double x = normalizedX[index];
        const double weight = weights[index];
        design.at<double>(index, 0) = x * x * weight;
        design.at<double>(index, 1) = x * weight;
        design.at<double>(index, 2) = weight;
        observations.at<double>(index) = y[index] * weight;
    }
    cv::Mat coefficients;
    if (!cv::solve(design, observations, coefficients, cv::DECOMP_SVD)) {
        throw std::runtime_error("二次曲线拟合失败");
    }
    return {
        coefficients.at<double>(0),
        coefficients.at<double>(1),
        coefficients.at<double>(2),
    };
}

double evaluate(const std::array<double, 3>& coefficients, double x) {
    return coefficients[0] * x * x + coefficients[1] * x + coefficients[2];
}

}  // namespace

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("无法计算空序列的中位数");
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 != 0) {
        return upper;
    }
    const double lower = *std::max_element(values.begin(), values.begin() + middle);
    return (lower + upper) / 2.0;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        throw std::invalid_argument("无法计算空序列的百分位数");
    }
    if (q < 0.0 || q > 100.0) {
        throw std::invalid_argument("百分位数必须在 0 到 100 之间");
    }
    std::sort(values.begin(), values.end());
    const double rank = (values.size() - 1) * q / 100.0;
    const auto lowerIndex = static_cast<std::size_t>(std::floor(rank));
    const auto upperIndex = static_cast<std::size_t>(std::ceil(rank));
    if (lowerIndex == upperIndex) {
        return values[lowerIndex];
    }
    const double fraction = rank - lowerIndex;
    return values[lowerIndex] + (values[upperIndex] - values[lowerIndex]) * fraction;
}

std::vector<double> maskedValues(const cv::Mat& values, const cv::Mat& mask) {
    if (values.size() != mask.size() || mask.type() != CV_8U) {
        throw std::invalid_argument("数值图与掩膜尺寸或类型不匹配");
    }
    if (values.channels() != 1) {
        throw std::invalid_argument("仅支持单通道数值图");
    }

    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(cv::countNonZero(mask)));
    for (int y = 0; y < values.rows; ++y) {
        const auto* maskRow = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            switch (values.depth()) {
                case CV_8U:
                    result.push_back(values.ptr<std::uint8_t>(y)[x]);
                    break;
                case CV_16S:
                    result.push_back(values.ptr<std::int16_t>(y)[x]);
                    break;
                case CV_32F:
                    result.push_back(values.ptr<float>(y)[x]);
                    break;
                case CV_64F:
                    result.push_back(values.ptr<double>(y)[x]);
                    break;
                default:
                    throw std::invalid_argument("不支持的数值图深度");
            }
        }
    }
    return result;
}

QuadraticFit robustQuadraticFit(const std::vector<cv::Point2d>& points, int iterations) {
    if (points.size() < 3) {
        throw std::invalid_argument("二次曲线拟合至少需要 3 个点");
    }
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(points.size());
    y.reserve(points.size());
    for (const auto& point : points) {
        x.push_back(point.x);
        y.push_back(point.y);
    }
    const double xCenter = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    const auto [minimum, maximum] = std::minmax_element(x.begin(), x.end());
    const double xScale = std::max((*maximum - *minimum) / 2.0, 1.0);

    std::vector<double> normalizedX(x.size());
    std::transform(x.begin(), x.end(), normalizedX.begin(), [&](double value) {
        return (value - xCenter) / xScale;
    });
    std::vector<double> weights(points.size(), 1.0);
    auto coefficients = solveQuadratic(normalizedX, y, weights);

    std::vector<double> fittedY(points.size());
    std::vector<double> residuals(points.size());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t index = 0; index < points.size(); ++index) {
            fittedY[index] = evaluate(coefficients, normalizedX[index]);
            residuals[index] = y[index] - fittedY[index];
        }
        const double residualMedian = median(residuals);
        std::vector<double> absoluteDeviations(residuals.size());
        std::transform(
            residuals.begin(), residuals.end(), absoluteDeviations.begin(),
            [&](double value) { return std::abs(value - residualMedian); });
        const double robustScale = std::max(1.0, 1.4826 * median(absoluteDeviations));
        const double cutoff = 1.5 * robustScale;
        std::fill(weights.begin(), weights.end(), 1.0);
        for (std::size_t index = 0; index < residuals.size(); ++index) {
            const double deviation = std::abs(residuals[index] - residualMedian);
            if (deviation > cutoff) {
                weights[index] = cutoff / deviation;
            }
        }
        coefficients = solveQuadratic(normalizedX, y, weights);
    }

    for (std::size_t index = 0; index < points.size(); ++index) {
        fittedY[index] = evaluate(coefficients, normalizedX[index]);
        residuals[index] = y[index] - fittedY[index];
    }
    return {coefficients, std::move(fittedY), std::move(residuals)};
}

}  // namespace tube_defect::detail
