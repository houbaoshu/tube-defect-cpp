#include "tube_defect/models.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#include <opencv2/imgcodecs.hpp>

namespace tube_defect {
namespace {

double roundTo(double value, int digits) {
    const double scale = std::pow(10.0, digits);
    return std::nearbyint(value * scale) / scale;
}

nlohmann::ordered_json boundingBoxJson(const BoundingBox& box) {
    return {
        {"x", box.x},
        {"y", box.y},
        {"width", box.width},
        {"height", box.height},
    };
}

nlohmann::ordered_json featuresJson(const DefectFeatures& features) {
    return std::visit(
        [](const auto& value) -> nlohmann::ordered_json {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, InterfaceFeatures>) {
                return {
                    {"shape", value.shape},
                    {"arc_point_count", value.arcPointCount},
                    {"rmse_px", roundTo(value.rmsePx, 3)},
                    {"rmse_tolerance_px", roundTo(value.rmseTolerancePx, 3)},
                    {"p95_deviation_px", roundTo(value.p95DeviationPx, 3)},
                    {"p95_tolerance_px", roundTo(value.p95TolerancePx, 3)},
                    {"max_deviation_px", roundTo(value.maxDeviationPx, 3)},
                    {"quadratic_coefficients", value.quadraticCoefficients},
                };
            } else if constexpr (std::is_same_v<T, ColorSpotFeatures>) {
                return {
                    {"shape", value.shape},
                    {"area_px", roundTo(value.areaPx, 2)},
                    {"aspect_ratio", roundTo(value.aspectRatio, 3)},
                    {"fill_ratio", roundTo(value.fillRatio, 3)},
                    {"circularity", roundTo(value.circularity, 3)},
                    {"mean_hsv",
                     {roundTo(value.meanHsv[0], 2),
                      roundTo(value.meanHsv[1], 2),
                      roundTo(value.meanHsv[2], 2)}},
                    {"local_contrast_gray", roundTo(value.localContrastGray, 2)},
                };
            } else if constexpr (std::is_same_v<T, ScratchFeatures>) {
                return {
                    {"shape", value.shape},
                    {"edge_area_px", value.edgeAreaPx},
                    {"angle_to_tube_axis_degrees",
                     roundTo(value.angleToTubeAxisDegrees, 2)},
                    {"length_px", roundTo(value.lengthPx, 2)},
                    {"width_px", roundTo(value.widthPx, 2)},
                    {"aspect_ratio", roundTo(value.aspectRatio, 3)},
                };
            } else {
                return {
                    {"shape", value.shape},
                    {"area_px", value.areaPx},
                    {"equivalent_diameter_px", roundTo(value.equivalentDiameterPx, 2)},
                    {"fill_ratio", roundTo(value.fillRatio, 3)},
                    {"circularity", roundTo(value.circularity, 3)},
                    {"median_gray", roundTo(value.medianGray, 2)},
                    {"background_median_gray", roundTo(value.backgroundMedianGray, 2)},
                    {"local_contrast_gray", roundTo(value.localContrastGray, 2)},
                };
            }
        },
        features);
}

}  // namespace

int BoundingBox::right() const noexcept {
    return x + width - 1;
}

int BoundingBox::bottom() const noexcept {
    return y + height - 1;
}

cv::Point BoundingBox::center() const noexcept {
    return {x + width / 2, y + height / 2};
}

std::string describePosition(cv::Point center, int imageWidth, int imageHeight) {
    const std::string horizontal = center.x < imageWidth / 3.0
                                       ? "左"
                                       : center.x > imageWidth * 2.0 / 3.0 ? "右" : "中";
    const std::string vertical = center.y < imageHeight / 3.0
                                     ? "上"
                                     : center.y > imageHeight * 2.0 / 3.0 ? "下" : "中";
    if (horizontal == "中" && vertical == "中") {
        return "图像中部";
    }
    if (horizontal == "中") {
        return "图像" + vertical + "部";
    }
    if (vertical == "中") {
        return "图像" + horizontal + "部";
    }
    return "图像" + horizontal + vertical + "部";
}

std::string_view toString(DetectionStatus status) noexcept {
    return status == DetectionStatus::defect ? "defect" : "normal";
}

std::string_view toString(SceneType sceneType) noexcept {
    switch (sceneType) {
        case SceneType::blueInterfaceTube:
            return "blue_interface_tube";
        case SceneType::markedClearTube:
            return "marked_clear_tube";
        case SceneType::clearTubeColorInspection:
            return "clear_tube_color_inspection";
        case SceneType::clearTube:
        default:
            return "clear_tube";
    }
}

std::string_view toString(DefectType defectType) noexcept {
    switch (defectType) {
        case DefectType::interfaceDistortion:
            return "interface_distortion";
        case DefectType::colorSpot:
            return "color_spot";
        case DefectType::scratch:
            return "scratch";
        case DefectType::hole:
        default:
            return "hole";
    }
}

std::string_view defectName(DefectType defectType) noexcept {
    switch (defectType) {
        case DefectType::interfaceDistortion:
            return "接口扭曲";
        case DefectType::colorSpot:
            return "黄褐色色斑";
        case DefectType::scratch:
            return "斜向划痕";
        case DefectType::hole:
        default:
            return "孔洞";
    }
}

nlohmann::ordered_json DetectionReport::toJson() const {
    nlohmann::ordered_json defectsJson = nlohmann::ordered_json::array();
    for (const auto& finding : defects) {
        const cv::Point center = finding.bbox.center();
        defectsJson.push_back({
            {"type", toString(finding.type)},
            {"name", defectName(finding.type)},
            {"confidence", roundTo(finding.confidence, 4)},
            {"location",
             {
                 {"bbox", boundingBoxJson(finding.bbox)},
                 {"centroid", {{"x", center.x}, {"y", center.y}}},
                 {"normalized_centroid",
                  {{"x", roundTo(static_cast<double>(center.x) / width, 6)},
                   {"y", roundTo(static_cast<double>(center.y) / height, 6)}}},
                 {"position", describePosition(center, width, height)},
             }},
            {"description", finding.description},
            {"features", featuresJson(finding.features)},
        });
    }

    const std::string primaryType = defects.empty()
                                        ? "normal"
                                        : std::string(toString(defects.front().type));
    const std::string primaryName = defects.empty()
                                        ? "正常"
                                        : std::string(defectName(defects.front().type));
    return {
        {"image", {{"path", imagePath}, {"width", width}, {"height", height}}},
        {"status", toString(status)},
        {"scene_type", toString(sceneType)},
        {"summary", summary},
        {"primary_defect", {{"type", primaryType}, {"name", primaryName}}},
        {"defect_count", defects.size()},
        {"defects", std::move(defectsJson)},
        {"warnings", warnings},
    };
}

std::pair<std::filesystem::path, std::filesystem::path> DetectionResult::save(
    const std::filesystem::path& outputDir) const {
    std::filesystem::create_directories(outputDir);
    const auto reportPath = outputDir / "result.json";
    const auto annotatedPath = outputDir / "annotated.png";

    std::ofstream reportFile(reportPath, std::ios::binary);
    if (!reportFile) {
        throw std::runtime_error("无法写入 JSON 报告: " + reportPath.string());
    }
    reportFile << report.toJson().dump(2) << '\n';
    reportFile.close();

    if (annotatedImage.empty() || !cv::imwrite(annotatedPath.string(), annotatedImage)) {
        throw std::runtime_error("无法保存标注图: " + annotatedPath.string());
    }

    if (!evidenceImages.empty()) {
        const auto evidenceDir = outputDir / "evidence";
        std::filesystem::create_directories(evidenceDir);
        for (const auto& [name, image] : evidenceImages) {
            const auto path = evidenceDir / (name + ".png");
            if (image.empty() || !cv::imwrite(path.string(), image)) {
                throw std::runtime_error("无法保存证据图: " + path.string());
            }
        }
    }
    return {reportPath, annotatedPath};
}

}  // namespace tube_defect
