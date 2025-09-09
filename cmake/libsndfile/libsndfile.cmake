include(ExternalProject)

ExternalProject_Add(libsndfile
    URL      https://github.com/libsndfile/libsndfile/archive/refs/tags/1.2.2.tar.gz
    URL_HASH SHA256=ffe12ef8add3eaca876f04087734e6e8e029350082f3251f565fa9da55b52121
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_PROGRAMS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_TESTING=OFF
)


ExternalProject_Get_property(libsndfile SOURCE_DIR)
set(libsndfile_SOURCE_DIR ${SOURCE_DIR})
set(SOURCE_DIR)

ExternalProject_Get_property(libsndfile INSTALL_DIR)
set(libsndfile_INSTALL_DIR ${INSTALL_DIR})
set(INSTALL_DIR)

set(libsndfile_INCLUDE_DIR ${libsndfile_INSTALL_DIR}/include)
set(libsndfile_LIBRARY_DIR ${libsndfile_INSTALL_DIR}/lib)
set(libsndfile_BINARY_DIR ${libsndfile_INSTALL_DIR}/bin)
