project("ParametricGaussRecon")
cmake_minimum_required(VERSION 3.0)

find_package(ZLIB REQUIRED)

set(SOURCES
    src/Cube.cpp
    src/Geometry.cpp
    src/MarchingCubes.cpp
    src/Mesh.cpp
    src/Octnode.cpp
    src/Octree.cpp
    src/ply.cpp
    src/plyfile.cpp)

add_executable(PGRExportQuery src/PGRExportQuery.cpp ${SOURCES})
target_compile_features(PGRExportQuery PRIVATE cxx_std_17)
target_include_directories(PGRExportQuery PRIVATE "${PROJECT_SOURCE_DIR}/third-party/cnpy/" "${PROJECT_SOURCE_DIR}/third-party/CLI/")
target_link_directories(PGRExportQuery PRIVATE "${PROJECT_SOURCE_DIR}/third-party/cnpy/")
target_link_libraries(PGRExportQuery PRIVATE cnpy)
set_target_properties(PGRExportQuery PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/apps/")

add_executable(PGRLoadQuery src/PGRLoadQuery.cpp ${SOURCES})
target_compile_features(PGRLoadQuery PRIVATE cxx_std_17)
target_include_directories(PGRLoadQuery PRIVATE "${PROJECT_SOURCE_DIR}/third-party/cnpy/" "${PROJECT_SOURCE_DIR}/third-party/CLI/")
target_link_directories(PGRLoadQuery PRIVATE "${PROJECT_SOURCE_DIR}/third-party/cnpy/")
target_link_libraries(PGRLoadQuery PRIVATE cnpy)
set_target_properties(PGRLoadQuery PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/apps/")
