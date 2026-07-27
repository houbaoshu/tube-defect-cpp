#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "tube_defect/detection.hpp"

namespace {

struct Options {
    std::filesystem::path image{"images"};
    std::optional<std::uint32_t> seed;
    std::optional<std::filesystem::path> outputDir;
    bool printJson{};
    bool help{};
};

void printUsage(const char* program) {
    std::cout
        << "用法: " << program << " [图片或目录] [选项]\n\n"
        << "随机选择或指定一张试管图片，判断缺陷类型、位置和特征。\n\n"
        << "选项:\n"
        << "  --seed <整数>       目录随机选择时使用的种子\n"
        << "  --output-dir <目录> 结果目录；默认 output/detection/<图片名>\n"
        << "  --print-json        在终端打印完整 JSON 报告\n"
        << "  -h, --help          显示帮助\n";
}

Options parseArguments(int argc, char** argv) {
    Options options;
    bool hasImage = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            options.help = true;
        } else if (argument == "--print-json") {
            options.printJson = true;
        } else if (argument == "--seed") {
            if (++index >= argc) {
                throw std::invalid_argument("--seed 缺少整数参数");
            }
            const unsigned long value = std::stoul(argv[index]);
            options.seed = static_cast<std::uint32_t>(value);
        } else if (argument == "--output-dir") {
            if (++index >= argc) {
                throw std::invalid_argument("--output-dir 缺少目录参数");
            }
            options.outputDir = std::filesystem::path(argv[index]);
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("未知选项: " + argument);
        } else if (!hasImage) {
            options.image = argument;
            hasImage = true;
        } else {
            throw std::invalid_argument("只能指定一个图片或目录路径");
        }
    }
    return options;
}

std::string lowercase(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool isSupportedImage(const std::filesystem::path& path) {
    const std::string extension = lowercase(path.extension().string());
    return extension == ".bmp" || extension == ".png" || extension == ".jpg" ||
           extension == ".jpeg" || extension == ".tif" || extension == ".tiff";
}

std::filesystem::path chooseImage(
    const std::filesystem::path& path,
    const std::optional<std::uint32_t>& seed) {
    if (std::filesystem::is_regular_file(path)) {
        return path;
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("输入路径不存在: " + path.string());
    }
    std::vector<std::filesystem::path> candidates;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() && isSupportedImage(entry.path())) {
            candidates.push_back(entry.path());
        }
    }
    std::ranges::sort(candidates);
    if (candidates.empty()) {
        throw std::runtime_error("目录中没有支持的图片: " + path.string());
    }
    std::mt19937 generator(seed.value_or(std::random_device{}()));
    std::uniform_int_distribution<std::size_t> distribution(0, candidates.size() - 1);
    return candidates[distribution(generator)];
}

void printHumanReport(const nlohmann::ordered_json& report) {
    std::cout << "输入图片: " << report["image"]["path"].get<std::string>() << '\n';
    std::cout << "判定结果: " << report["primary_defect"]["name"].get<std::string>() << '\n';
    std::cout << "结论: " << report["summary"].get<std::string>() << '\n';
    std::size_t index = 1;
    for (const auto& defect : report["defects"]) {
        const auto& location = defect["location"];
        const auto& bbox = location["bbox"];
        std::cout << "缺陷 #" << index++ << ": " << defect["name"].get<std::string>()
                  << " | " << location["position"].get<std::string>()
                  << " | 中心=(" << location["centroid"]["x"] << ", "
                  << location["centroid"]["y"] << ")"
                  << " | bbox=(" << bbox["x"] << ", " << bbox["y"] << ", "
                  << bbox["width"] << ", " << bbox["height"] << ")\n";
        std::cout << "  特征: " << defect["features"].dump() << '\n';
    }
    for (const auto& warning : report["warnings"]) {
        std::cout << "警告: " << warning.get<std::string>() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        if (options.help) {
            printUsage(argv[0]);
            return 0;
        }
        const std::filesystem::path imagePath = chooseImage(options.image, options.seed);
        const tube_defect::DetectionResult result = tube_defect::analyzePath(imagePath);
        const std::filesystem::path outputDir = options.outputDir.value_or(
            std::filesystem::path("output") / "detection" / imagePath.stem());
        const auto [reportPath, annotatedPath] = result.save(outputDir);
        const auto report = result.report.toJson();
        if (options.printJson) {
            std::cout << report.dump(2) << '\n';
        } else {
            printHumanReport(report);
        }
        std::cout << "JSON 报告: " << std::filesystem::absolute(reportPath) << '\n';
        std::cout << "标注图片: " << std::filesystem::absolute(annotatedPath) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
