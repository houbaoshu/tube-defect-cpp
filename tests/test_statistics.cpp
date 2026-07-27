#include <iostream>
#include <vector>

#include "detail/statistics.hpp"
#include "test_support.hpp"

int main() {
    try {
        checkNear(tube_defect::detail::median({1, 9, 3}), 3.0, 1e-12);
        checkNear(tube_defect::detail::median({1, 9, 3, 5}), 4.0, 1e-12);
        checkNear(tube_defect::detail::percentile({0, 10, 20, 30}, 95), 28.5, 1e-12);

        std::vector<cv::Point2d> points;
        for (int x = -20; x <= 20; ++x) {
            double y = 2.0 * x * x - 3.0 * x + 5.0;
            if (x == 0) {
                y += 120.0;
            }
            points.emplace_back(x, y);
        }
        const auto fit = tube_defect::detail::robustQuadraticFit(points);
        CHECK_TRUE(fit.residuals.size() == points.size());
        CHECK_TRUE(std::abs(fit.residuals[20]) > 100.0);
        CHECK_TRUE(std::abs(fit.residuals.front()) < 1.0);
        std::cout << "statistics tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
