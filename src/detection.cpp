#include "tube_defect/detection.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "detail/opencv_geometry.hpp"
#include "detectors/color_spot.hpp"
#include "detectors/hole.hpp"
#include "detectors/interface.hpp"
#include "detectors/scratch.hpp"
#include "visualization.hpp"

namespace tube_defect {
namespace {

BoundingBox box(const cv::Rect& rectangle) {
    return {rectangle.x, rectangle.y, rectangle.width, rectangle.height};
}

DefectFinding interfaceFinding(const detectors::InterfaceCircleResult& result) {
    const cv::Rect rectangle = cv::boundingRect(result.contour);
    const double severity = std::max(
        result.rmse / result.rmseTolerance,
        result.p95Deviation / result.p95Tolerance);
    const double confidence = std::min(0.99, 0.72 + std::max(0.0, severity - 1.0) * 0.18);
    return {
        DefectType::interfaceDistortion,
        box(rectangle),
        confidence,
        "接口实际边界相对鲁棒二次弧线出现连续偏离，RMSE 与 P95 残差均超过允许阈值。",
        InterfaceFeatures{
            "接口弧线不平滑/局部错位",
            result.contour.total(),
            result.rmse,
            result.rmseTolerance,
            result.p95Deviation,
            result.p95Tolerance,
            result.maxDeviation,
            result.arcCoefficients,
        },
    };
}

std::vector<DefectFinding> spotFindings(const std::vector<detectors::ColorSpotResult>& items) {
    std::vector<DefectFinding> findings;
    findings.reserve(items.size());
    for (const auto& item : items) {
        const double confidence = std::min(
            0.98,
            0.70 + item.fillRatio * 0.18 + item.circularity * 0.10);
        findings.push_back({
            DefectType::colorSpot,
            box(item.bbox),
            confidence,
            "局部区域呈黄褐色、近圆形且相对周围表面更暗，符合污染或色斑特征。",
            ColorSpotFeatures{
                "近圆形色斑",
                item.area,
                item.aspectRatio,
                item.fillRatio,
                item.circularity,
                item.meanHsv,
                item.contrast,
            },
        });
    }
    return findings;
}

std::vector<DefectFinding> scratchFindings(const std::vector<detectors::ScratchResult>& items) {
    std::vector<DefectFinding> findings;
    findings.reserve(items.size());
    for (const auto& item : items) {
        const double confidence = std::min(0.98, 0.72 + std::min(item.aspectRatio, 4.0) * 0.05);
        findings.push_back({
            DefectType::scratch,
            box(item.bbox),
            confidence,
            "表面出现与管轴成明显夹角的细长连续边缘，符合斜向划痕特征。",
            ScratchFeatures{
                "细长斜向线状",
                item.area,
                item.angleDegrees,
                item.length,
                item.width,
                item.aspectRatio,
            },
        });
    }
    return findings;
}

std::vector<DefectFinding> holeFindings(const std::vector<detectors::HoleResult>& items) {
    std::vector<DefectFinding> findings;
    findings.reserve(items.size());
    for (const auto& item : items) {
        const double confidence = std::min(0.98, 0.70 + std::min(item.contrast, 80.0) / 400.0);
        findings.push_back({
            DefectType::hole,
            box(item.bbox),
            confidence,
            "局部为小尺寸暗色闭合区域，中心灰度显著低于邻域，符合孔洞特征。",
            HoleFeatures{
                "小型暗色闭合斑",
                item.area,
                item.equivalentDiameter,
                item.fillRatio,
                item.circularity,
                item.medianGray,
                item.backgroundMedianGray,
                item.contrast,
            },
        });
    }
    return findings;
}

void append(std::vector<DefectFinding>& destination, std::vector<DefectFinding> source) {
    destination.insert(
        destination.end(),
        std::make_move_iterator(source.begin()),
        std::make_move_iterator(source.end()));
}

std::string findingsSummary(const std::vector<DefectFinding>& findings) {
    std::vector<std::string> names;
    for (const auto& finding : findings) {
        const std::string name(defectName(finding.type));
        if (std::ranges::find(names, name) == names.end()) {
            names.push_back(name);
        }
    }
    std::string joined;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (index > 0) {
            joined += "、";
        }
        joined += names[index];
    }
    return "检测到" + joined + "，共 " + std::to_string(findings.size()) + " 处。";
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

DetectionResult analyzeImage(const cv::Mat& image, std::string imagePath) {
    validateImage(image);
    std::vector<std::string> warnings;
    std::map<std::string, cv::Mat> evidence;
    std::vector<DefectFinding> findings;
    SceneType sceneType = SceneType::clearTube;
    cv::Mat baseVisualization = image;
    std::string normalSummary = "未检测到已配置的缺陷。";

    if (detectors::hasFilledBlueTube(image)) {
        sceneType = SceneType::blueInterfaceTube;
        try {
            const auto interface = detectors::detectInterfaceCircle(image);
            baseVisualization = interface.visualization;
            evidence.emplace("interface_edges", interface.edges);
            evidence.emplace("interface_arc_mask", interface.candidateMask);
            if (interface.isDistorted) {
                findings.push_back(interfaceFinding(interface));
            } else {
                normalSummary = "蓝色软管接口弧线平滑，拟合 RMSE 与 P95 残差均在阈值内。";
            }
        } catch (const std::exception& error) {
            warnings.push_back(std::string("接口检测未完成: ") + error.what());
        }
    } else {
        const auto marker = detectors::findLargeOrangeMarker(image);
        if (marker) {
            sceneType = SceneType::markedClearTube;
            auto scratches = detectors::detectScratches(image, marker);
            evidence.emplace("scratch_mask", scratches.mask);
            append(findings, scratchFindings(scratches.findings));
            if (findings.empty()) {
                normalSummary = "检测到带卡环透明管，但未发现满足阈值的斜向划痕。";
            }
        } else {
            auto [spots, spotMask] = detectors::detectColorSpots(image);
            if (!spots.empty()) {
                sceneType = SceneType::clearTubeColorInspection;
                evidence.emplace("color_spot_mask", spotMask);
                append(findings, spotFindings(spots));
            } else {
                auto holes = detectors::detectHoles(image);
                evidence.emplace("hole_mask", holes.mask);
                append(findings, holeFindings(holes.findings));
            }
        }
    }

    const DetectionStatus status = findings.empty()
                                       ? DetectionStatus::normal
                                       : DetectionStatus::defect;
    DetectionReport report{
        std::move(imagePath),
        image.cols,
        image.rows,
        status,
        sceneType,
        findings.empty() ? normalSummary : findingsSummary(findings),
        findings,
        warnings,
    };
    cv::Mat annotated = annotate(baseVisualization, findings, status);
    return {std::move(report), std::move(annotated), std::move(evidence)};
}

DetectionResult analyzePath(const std::filesystem::path& imagePath) {
    const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("无法读取图片: " + imagePath.string());
    }
    return analyzeImage(image, imagePath.string());
}

}  // namespace tube_defect
