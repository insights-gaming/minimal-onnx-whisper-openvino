include(ExternalProject)

ExternalProject_Add(fftw
    URL      https://fftw.org/fftw-3.3.10.tar.gz
    URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DENABLE_FLOAT=ON
        -DENABLE_LONG_DOUBLE=ON
)

ExternalProject_Get_property(fftw SOURCE_DIR)
set(fftw_SOURCE_DIR ${SOURCE_DIR})
set(SOURCE_DIR)

ExternalProject_Get_property(fftw INSTALL_DIR)
set(fftw_INSTALL_DIR ${INSTALL_DIR})
set(INSTALL_DIR)

set(fftw_INCLUDE_DIR ${fftw_INSTALL_DIR}/include)
set(fftw_LIBRARY_DIR ${fftw_INSTALL_DIR}/lib)
set(fftw_BINARY_DIR ${fftw_INSTALL_DIR}/bin)
