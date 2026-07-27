#include <iostream>

#include "test_support.hpp"
#include "tube_defect/models.hpp"

int main() {
    try {
        const tube_defect::BoundingBox box{10, 20, 11, 9};
        CHECK_TRUE(box.center() == cv::Point(15, 24));
        CHECK_TRUE(box.right() == 20);
        CHECK_TRUE(box.bottom() == 28);
        CHECK_TRUE(tube_defect::describePosition({50, 50}, 300, 300) == "图像左上部");
        CHECK_TRUE(tube_defect::describePosition({150, 150}, 300, 300) == "图像中部");
        CHECK_TRUE(tube_defect::describePosition({250, 150}, 300, 300) == "图像右部");

        const tube_defect::DetectionReport report{
            "sample.bmp",
            300,
            200,
            tube_defect::DetectionStatus::normal,
            tube_defect::SceneType::clearTube,
            "未检测到已配置的缺陷。",
            {},
            {},
        };
        const auto json = report.toJson();
        CHECK_TRUE(json["status"] == "normal");
        CHECK_TRUE(json["primary_defect"]["type"] == "normal");
        CHECK_TRUE(json["defect_count"] == 0);
        std::cout << "model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
