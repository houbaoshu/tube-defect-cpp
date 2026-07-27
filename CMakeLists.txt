cmake_minimum_required(VERSION 3.24)

project(tube_defect_detection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(tube_defect_detection
    src/detection.cpp
    src/models.cpp
    src/visualization.cpp
    src/detail/statistics.cpp
    src/detectors/color_spot.cpp
    src/detectors/hole.cpp
    src/detectors/interface.cpp
    src/detectors/scratch.cpp
)
add_library(tube_defect::detection ALIAS tube_defect_detection)

target_include_directories(tube_defect_detection
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(tube_defect_detection
    PUBLIC
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
)
target_compile_features(tube_defect_detection PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(tube_defect_detection PRIVATE /W4 /permissive-)
else()
    target_compile_options(tube_defect_detection PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(tube-defect-detection src/cli/main.cpp)
target_link_libraries(tube-defect-detection PRIVATE tube_defect::detection)

include(GNUInstallDirs)
install(TARGETS tube_defect_detection tube-defect-detection
    EXPORT tube_defect_detection_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT tube_defect_detection_targets
    FILE tube_defect_detection_targets.cmake
    NAMESPACE tube_defect::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tube_defect_detection
)

include(CTest)
if(BUILD_TESTING)
    function(add_tube_defect_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE tube_defect::detection)
        target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_compile_definitions(${name} PRIVATE TUBE_DEFECT_IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/images")
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_tube_defect_test(test_models tests/test_models.cpp)
    add_tube_defect_test(test_statistics tests/test_statistics.cpp)
    add_tube_defect_test(test_interface tests/test_interface.cpp)
    add_tube_defect_test(test_detection_pipeline tests/test_detection_pipeline.cpp)

    set(CLI_TEST_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-output/cli")
    add_test(
        NAME test_cli
        COMMAND ${CMAKE_COMMAND}
            -DCLI=$<TARGET_FILE:tube-defect-detection>
            -DIMAGE=${CMAKE_CURRENT_SOURCE_DIR}/images/02_01.bmp
            -DOUTPUT_DIR=${CLI_TEST_OUTPUT_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.cmake
    )
endif()
