include(FetchContent)

set(openvino_VERSION "2025.3")
set(openvino_BUILD "19807")
set(openvino_PATCH "44526285f24")
set(openvino_URL  "https://storage.openvinotoolkit.org/repositories/openvino/packages/${openvino_VERSION}/windows/openvino_toolkit_windows_${openvino_VERSION}.0.${openvino_BUILD}.${openvino_PATCH}_x86_64.zip")
set(openvino_HASH "SHA256=05685c652e85f92ad17572ec2800ea6d0b96c9b7ff645299ad2ba09d1afb17b4")

FetchContent_Declare(openvino
    URL      ${openvino_URL}
    URL_HASH ${openvino_HASH}
)

FetchContent_MakeAvailable(openvino)

set(openvino_INSTALL_DIR "${openvino_SOURCE_DIR}/runtime")
set(openvino_LIBRARY_DIR "${openvino_INSTALL_DIR}/lib/intel64/Release")
set(openvino_INCLUDE_DIR "${openvino_INSTALL_DIR}/include")
set(openvino_BINARY_DIR "${openvino_INSTALL_DIR}/bin/intel64/Release")
