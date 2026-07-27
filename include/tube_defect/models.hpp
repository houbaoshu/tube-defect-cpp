#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

namespace tube_defect {

struct BoundingBox {
    int x{};
    int y{};
    int width{};
    int height{};

    [[nodiscard]] int right() const noexcept;
    [[nodiscard]] int bottom() const noexcept;
    [[nodiscard]] cv::Point center() const noexcept;
};

enum class DetectionStatus { normal, defect };
enum class SceneType {
    clearTube,
    blueInterfaceTube,
    markedClearTube,
    clearTubeColorInspection,
};
enum class DefectType { interfaceDistortion, colorSpot, scratch, hole };

struct InterfaceFeatures {
    std::string shape{"接口弧线不平滑/局部错位"};
    std::size_t arcPointCount{};
    double rmsePx{};
    double rmseTolerancePx{};
    double p95DeviationPx{};
    double p95TolerancePx{};
    double maxDeviationPx{};
    std::array<double, 3> quadraticCoefficients{};
};

struct ColorSpotFeatures {
    std::string shape{"近圆形色斑"};
    double areaPx{};
    double aspectRatio{};
    double fillRatio{};
    double circularity{};
    std::array<double, 3> meanHsv{};
    double localContrastGray{};
};

struct ScratchFeatures {
    std::string shape{"细长斜向线状"};
    int edgeAreaPx{};
    double angleToTubeAxisDegrees{};
    double lengthPx{};
    double widthPx{};
    double aspectRatio{};
};

struct HoleFeatures {
    std::string shape{"小型暗色闭合斑"};
    int areaPx{};
    double equivalentDiameterPx{};
    double fillRatio{};
    double circularity{};
    double medianGray{};
    double backgroundMedianGray{};
    double localContrastGray{};
};

using DefectFeatures = std::variant<
    InterfaceFeatures,
    ColorSpotFeatures,
    ScratchFeatures,
    HoleFeatures>;

struct DefectFinding {
    DefectType type{};
    BoundingBox bbox{};
    double confidence{};
    std::string description;
    DefectFeatures features;
};

struct DetectionReport {
    std::string imagePath;
    int width{};
    int height{};
    DetectionStatus status{DetectionStatus::normal};
    SceneType sceneType{SceneType::clearTube};
    std::string summary;
    std::vector<DefectFinding> defects;
    std::vector<std::string> warnings;

    [[nodiscard]] nlohmann::ordered_json toJson() const;
};

struct DetectionResult {
    DetectionReport report;
    cv::Mat annotatedImage;
    std::map<std::string, cv::Mat> evidenceImages;

    [[nodiscard]] std::pair<std::filesystem::path, std::filesystem::path> save(
        const std::filesystem::path& outputDir) const;
};

[[nodiscard]] std::string describePosition(
    cv::Point center,
    int imageWidth,
    int imageHeight);
[[nodiscard]] std::string_view toString(DetectionStatus status) noexcept;
[[nodiscard]] std::string_view toString(SceneType sceneType) noexcept;
[[nodiscard]] std::string_view toString(DefectType defectType) noexcept;
[[nodiscard]] std::string_view defectName(DefectType defectType) noexcept;

}  // namespace tube_defect
