include(ExternalProject)

get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(${_IMPORT_PREFIX}/../openvino/openvino.cmake)

if(NOT onnxruntime_BUILD_TYPE)
    set(onnxruntime_BUILD_TYPE "Release")
endif()

set(onnxruntime_BASE_COMMAND <SOURCE_DIR>/build.bat
    --config ${onnxruntime_BUILD_TYPE}
    --skip_tests
    --parallel
    --compile_no_warning_as_error
    --build_shared_lib
    # --use_winml --enable_wcos
    # --use_dml
    --use_openvino AUTO:NPU,GPU,CPU
    --build_dir <BINARY_DIR>
)
set(onnxruntime_BASE_COMMAND
    COMMAND ${openvino_SOURCE_DIR}/setupvars.bat
    COMMAND ${onnxruntime_BASE_COMMAND}
)

ExternalProject_Add(onnxruntime
    URL      https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.22.2.tar.gz
    URL_HASH SHA256=6f82b949636df0c964cc6f1ef4f1b39b397dce456f92b204d87b46d258687b41
    CONFIGURE_COMMAND ${onnxruntime_BASE_COMMAND} --update
    BUILD_COMMAND ${onnxruntime_BASE_COMMAND} --build
    INSTALL_COMMAND cmake --install "<BINARY_DIR>/${onnxruntime_BUILD_TYPE}" --prefix <INSTALL_DIR>
)

ExternalProject_Get_property(onnxruntime SOURCE_DIR)
set(onnxruntime_SOURCE_DIR ${SOURCE_DIR})
set(SOURCE_DIR)

ExternalProject_Get_property(onnxruntime INSTALL_DIR)
set(onnxruntime_INSTALL_DIR ${INSTALL_DIR})
set(INSTALL_DIR)

set(onnxruntime_INCLUDE_DIR ${onnxruntime_INSTALL_DIR}/include)
set(onnxruntime_LIBRARY_DIR ${onnxruntime_INSTALL_DIR}/lib)
set(onnxruntime_BINARY_DIR ${onnxruntime_INSTALL_DIR}/bin)
