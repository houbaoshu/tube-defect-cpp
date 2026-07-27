#include <array>
#include <filesystem>
#include <iostream>
#include <string>

#include "test_support.hpp"
#include "tube_defect/detection.hpp"

namespace {

void checkBox(const nlohmann::ordered_json& json, const std::array<int, 4>& expected) {
    const auto& box = json["location"]["bbox"];
    checkNear(box["x"].get<int>(), expected[0], 2);
    checkNear(box["y"].get<int>(), expected[1], 2);
    checkNear(box["width"].get<int>(), expected[2], 2);
    checkNear(box["height"].get<int>(), expected[3], 2);
}

nlohmann::ordered_json analyze(const std::filesystem::path& imageDir, const std::string& name) {
    auto result = tube_defect::analyzePath(imageDir / name);
    CHECK_TRUE(result.annotatedImage.size() == cv::Size(3072, 2048));
    for (const auto& [evidenceName, image] : result.evidenceImages) {
        (void)evidenceName;
        CHECK_TRUE(image.size() == cv::Size(3072, 2048));
        CHECK_TRUE(image.type() == CV_8U);
    }
    return result.report.toJson();
}

}  // namespace

int main() {
    try {
        const std::filesystem::path imageDir = TUBE_DEFECT_IMAGE_DIR;
        const auto normal = analyze(imageDir, "01_01.bmp");
        CHECK_TRUE(normal["status"] == "normal");
        CHECK_TRUE(normal["primary_defect"]["type"] == "normal");
        CHECK_TRUE(normal["defect_count"] == 0);

        const auto interface = analyze(imageDir, "01_02.bmp");
        CHECK_TRUE(interface["primary_defect"]["type"] == "interface_distortion");
        CHECK_TRUE(interface["defect_count"] == 1);
        checkBox(interface["defects"][0], {1476, 855, 259, 35});
        CHECK_TRUE(interface["defects"][0]["features"]["rmse_px"] >
                   interface["defects"][0]["features"]["rmse_tolerance_px"]);

        const auto spot = analyze(imageDir, "02_01.bmp");
        CHECK_TRUE(spot["primary_defect"]["type"] == "color_spot");
        CHECK_TRUE(spot["defect_count"] == 1);
        checkBox(spot["defects"][0], {1898, 825, 31, 34});
        CHECK_TRUE(spot["defects"][0]["features"]["fill_ratio"] > 0.5);

        const auto scratch = analyze(imageDir, "3_01.bmp");
        CHECK_TRUE(scratch["primary_defect"]["type"] == "scratch");
        CHECK_TRUE(scratch["defect_count"] == 1);
        checkBox(scratch["defects"][0], {1857, 799, 150, 110});
        CHECK_TRUE(scratch["defects"][0]["features"]["angle_to_tube_axis_degrees"] > 15);
        CHECK_TRUE(scratch["defects"][0]["features"]["aspect_ratio"] > 2);

        const auto holes = analyze(imageDir, "4_01.bmp");
        CHECK_TRUE(holes["primary_defect"]["type"] == "hole");
        CHECK_TRUE(holes["defect_count"] == 2);
        checkBox(holes["defects"][0], {1748, 766, 14, 17});
        checkBox(holes["defects"][1], {1773, 692, 5, 16});
        for (const auto& hole : holes["defects"]) {
            CHECK_TRUE(hole["features"]["local_contrast_gray"] > 20);
        }
        std::cout << "pipeline tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
