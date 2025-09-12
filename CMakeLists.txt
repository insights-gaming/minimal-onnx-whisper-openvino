cmake_minimum_required(VERSION 3.15...3.31)
project(minimal_whisper)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime/onnxruntime.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/libsndfile/libsndfile.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftw/fftw.cmake)

message(STATUS "OpenVINO library directory: ${openvino_LIBRARY_DIR}")
message(STATUS "OpenVINO include directory: ${openvino_INCLUDE_DIR}")

message(STATUS "ONNX Runtime library directory: ${onnxruntime_LIBRARY_DIR}")
message(STATUS "ONNX Runtime include directory: ${onnxruntime_INCLUDE_DIR}")

# add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
# set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
# add_dependencies(${PROJECT_NAME} onnxruntime)

add_executable(${PROJECT_NAME}
    minimal_whisper.cc
)

add_dependencies(${PROJECT_NAME} onnxruntime)
add_dependencies(${PROJECT_NAME} libsndfile)
add_dependencies(${PROJECT_NAME} fftw)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${onnxruntime_INCLUDE_DIR}/onnxruntime
    ${libsndfile_INCLUDE_DIR}
    ${fftw_INCLUDE_DIR}
)
target_link_directories(${PROJECT_NAME} PRIVATE
    ${onnxruntime_LIBRARY_DIR}
    ${libsndfile_LIBRARY_DIR}
    ${fftw_LIBRARY_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    "onnxruntime.lib"
    "sndfile.lib"
    "fftw3.lib"
    "fftw3l.lib"
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)


set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${onnxruntime_BINARY_DIR}/onnxruntime.dll"
        "${onnxruntime_BINARY_DIR}/onnxruntime_providers_shared.dll"
        "${onnxruntime_LIBRARY_DIR}/onnxruntime_providers_openvino.dll"
        "${openvino_BINARY_DIR}/openvino.dll"
        "${openvino_INSTALL_DIR}/3rdparty/tbb/bin/tbb12.dll"
        "${libsndfile_BINARY_DIR}/sndfile.dll"
        "${fftw_BINARY_DIR}/fftw3.dll"
        "${fftw_BINARY_DIR}/fftw3l.dll"
        "${OUTPUT_DIR}"
)
