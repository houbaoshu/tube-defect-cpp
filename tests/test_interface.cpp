#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "detectors/interface.hpp"
#include "test_support.hpp"

int main() {
    try {
        const std::filesystem::path imageDir = TUBE_DEFECT_IMAGE_DIR;
        const cv::Mat normal = cv::imread((imageDir / "01_01.bmp").string());
        const cv::Mat distorted = cv::imread((imageDir / "01_02.bmp").string());
        CHECK_TRUE(!normal.empty());
        CHECK_TRUE(!distorted.empty());
        CHECK_TRUE(tube_defect::detectors::hasFilledBlueTube(normal));
        CHECK_TRUE(tube_defect::detectors::hasFilledBlueTube(distorted));

        const auto normalResult = tube_defect::detectors::detectInterfaceCircle(normal);
        CHECK_TRUE(!normalResult.isDistorted);
        CHECK_TRUE(normalResult.rmse < normalResult.rmseTolerance);
        CHECK_TRUE(normalResult.p95Deviation < normalResult.p95Tolerance);

        const auto defectResult = tube_defect::detectors::detectInterfaceCircle(distorted);
        CHECK_TRUE(defectResult.isDistorted);
        CHECK_TRUE(defectResult.rmse > defectResult.rmseTolerance);
        CHECK_TRUE(defectResult.p95Deviation > defectResult.p95Tolerance);
        CHECK_TRUE(defectResult.contour.total() == 259);
        checkNear(defectResult.rmse, 5.816, 0.05);
        checkNear(defectResult.p95Deviation, 12.662, 0.10);
        std::cout << "interface tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
