include(ExternalProject)

ExternalProject_Add(fftw3
    URL      https://fftw.org/fftw-3.3.10.tar.gz
    URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)

ExternalProject_Get_property(fftw3 SOURCE_DIR)
set(fftw3_SOURCE_DIR ${SOURCE_DIR})
set(SOURCE_DIR)

ExternalProject_Get_property(fftw3 INSTALL_DIR)
set(fftw3_INSTALL_DIR ${INSTALL_DIR})
set(INSTALL_DIR)

set(fftw3_INCLUDE_DIR ${fftw3_INSTALL_DIR}/include)
set(fftw3_LIBRARY_DIR ${fftw3_INSTALL_DIR}/lib)
set(fftw3_BINARY_DIR ${fftw3_INSTALL_DIR}/bin)


ExternalProject_Add(fftw3l
    URL      https://fftw.org/fftw-3.3.10.tar.gz
    URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DENABLE_LONG_DOUBLE=ON
)

ExternalProject_Get_property(fftw3l SOURCE_DIR)
set(fftw3l_SOURCE_DIR ${SOURCE_DIR})
set(SOURCE_DIR)

ExternalProject_Get_property(fftw3l INSTALL_DIR)
set(fftw3l_INSTALL_DIR ${INSTALL_DIR})
set(INSTALL_DIR)

set(fftw3l_INCLUDE_DIR ${fftw3l_INSTALL_DIR}/include)
set(fftw3l_LIBRARY_DIR ${fftw3l_INSTALL_DIR}/lib)
set(fftw3l_BINARY_DIR ${fftw3l_INSTALL_DIR}/bin)
