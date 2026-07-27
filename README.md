# Tube defect detection（C++ / OpenCV）

使用 C++20 和 OpenCV 对试管原始整图进行可解释的缺陷检测。程序自动识别当前检测场景，输出缺陷类型、原图坐标、置信度、量化特征、标注图和各检测器的证据图。

当前支持：

- **接口扭曲**：接口弧线拟合 RMSE、P95 残差、最大偏差
- **黄褐色色斑**：HSV、面积、圆度、填充率、局部对比度
- **斜向划痕**：相对管轴角度、长度、宽度、长宽比、边缘面积
- **孔洞**：面积、等效直径、圆度、中心灰度、邻域对比度
- **正常**：所有已配置检测器均未达到缺陷阈值

检测直接使用原始整图，不需要手工截取 ROI；结果坐标始终相对于输入原图左上角。

## 依赖

- CMake 3.24+
- 支持 C++20 的编译器
- OpenCV 4 或 5（`core`、`imgproc`、`imgcodecs`）
- nlohmann-json 3.x
- Ninja（使用仓库预设时）

### macOS / Homebrew

```bash
brew install cmake ninja opencv nlohmann-json
cmake --preset macos-homebrew
cmake --build --preset macos-homebrew
ctest --preset macos-homebrew --output-on-failure
```

该预设使用 `/opt/homebrew/opt/opencv` 和 `/opt/homebrew/opt/nlohmann-json`。Intel Mac 或自定义 Homebrew 路径可使用通用配置并显式传入前缀：

```bash
cmake -S . -B build/release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix opencv);$(brew --prefix nlohmann-json)"
cmake --build build/release
ctest --test-dir build/release --output-on-failure
```

### Linux

安装发行版提供的 OpenCV、nlohmann-json、CMake 和 Ninja 开发包后：

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default --output-on-failure
```

如果依赖安装在非系统目录，通过 `CMAKE_PREFIX_PATH` 或 `OpenCV_DIR` 指定位置。

## 快速使用

指定一张图片：

```bash
./build/macos-homebrew/tube-defect-detection images/4_01.bmp
```

传入目录时会随机选择一张支持的图片；同一个 C++ 版本中，`--seed` 可复现选择结果：

```bash
./build/macos-homebrew/tube-defect-detection images --seed 42
```

打印完整 JSON：

```bash
./build/macos-homebrew/tube-defect-detection images/3_01.bmp --print-json
```

指定输出目录：

```bash
./build/macos-homebrew/tube-defect-detection \
  images/02_01.bmp \
  --output-dir output/my-detection
```

完整参数：

```text
用法: tube-defect-detection [图片或目录] [选项]

  --seed <整数>       目录随机选择时使用的种子
  --output-dir <目录> 结果目录；默认 output/detection/<图片名>
  --print-json        在终端打印完整 JSON 报告
  -h, --help          显示帮助
```

## 输出

默认保存在 `output/detection/<图片名>/`：

```text
result.json
annotated.png
evidence/
├── interface_edges.png
├── interface_arc_mask.png
├── color_spot_mask.png
├── scratch_mask.png
└── hole_mask.png
```

只会生成当前场景实际使用到的证据图。`result.json` 保持原 Python 版本的字段契约，包括：

- 图片路径、宽度、高度
- `status` 和 `scene_type`
- 主缺陷类型与中文名称
- bbox、中心点、归一化中心点和九宫格方位
- 缺陷置信度、说明和类型专用特征
- 检测警告

## C++ API

```cpp
#include <iostream>

#include "tube_defect/detection.hpp"

int main() {
    auto result = tube_defect::analyzePath("images/02_01.bmp");
    std::cout << result.report.toJson().dump(2) << '\n';
    result.save("output/from-cpp");
}
```

在 CMake 工程中直接引入本项目：

```cmake
add_subdirectory(path/to/tube-defect-cpp)
target_link_libraries(your_target PRIVATE tube_defect::detection)
```

内存图像可调用：

```cpp
cv::Mat bgr = /* CV_8UC3 */;
auto result = tube_defect::analyzeImage(bgr, "camera-frame");
```

输入必须是非空 `CV_8UC3` BGR 图像。

## 代码结构

```text
include/tube_defect/
├── detection.hpp              # 公共分析 API
└── models.hpp                 # 报告、缺陷和特征模型
src/
├── cli/main.cpp               # 命令行入口
├── detection.cpp              # 场景路由和统一检测管线
├── models.cpp                 # JSON、位置描述和结果保存
├── visualization.cpp          # 原图标注
├── detail/statistics.cpp      # 百分位数、掩膜统计、鲁棒二次拟合
└── detectors/
    ├── interface.cpp          # 接口扭曲
    ├── color_spot.cpp         # 黄褐色色斑
    ├── scratch.cpp            # 斜向划痕
    └── hole.cpp               # 孔洞
tests/                         # CTest 单元、回归和 CLI 测试
```

生产模型使用枚举和每类缺陷专用的特征结构；检测器返回类型化结果，JSON 只在模型层生成。NumPy 中位数、线性插值百分位数、鲁棒二次拟合和 PCA 等行为均有对应的 C++ 实现或 OpenCV 等价实现。

## 回归基线

五张现有样例的预期主判定：

| 图片 | 判定 | 数量 |
|---|---|---:|
| `01_01.bmp` | 正常 | 0 |
| `01_02.bmp` | 接口扭曲 | 1 |
| `02_01.bmp` | 黄褐色色斑 | 1 |
| `3_01.bmp` | 斜向划痕 | 1 |
| `4_01.bmp` | 孔洞 | 2 |

运行全部测试：

```bash
ctest --preset macos-homebrew --output-on-failure
```

测试覆盖数据模型、统计函数、正常/异常接口、五张整图检测管线以及 CLI 保存行为。当前 C++ 实现已与 Python 0.2.0 基线逐图比较，分类、场景、缺陷数量、位置和序列化特征一致；接口拟合系数仅存在浮点求解器量级的误差。

## 迁移状态

`tube-defect-opencv-0.2.0/` 和 `tube-defect-opencv-0.2.0-source.zip` 暂时保留，作为本次并行迁移的 Python 行为基线。C++ 版本通过业务验收后，可在单独步骤中移除旧 Python 源码、锁文件和压缩包；本阶段未改写或删除它们。

## 适用范围

当前实现是按现有相机、照明、检测台和试管结构标定的传统视觉方案，不是经过大规模数据训练的通用语义模型。更换相机、曝光、背景、管体颜色或缺陷尺度后，应使用新的正常/缺陷样本重新标定阈值，并补充回归测试。
