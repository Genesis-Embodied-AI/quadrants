# ./quadrants/util/CMakeLists.txt

add_library(quadrants_util)
target_sources(quadrants_util
  PRIVATE
    bit.cpp
    file_sequence_writer.cpp
    lang_util.cpp
    offline_cache.cpp
    short_name.cpp
    str.cpp
    testing.cpp
    zip.cpp
  )

target_include_directories(quadrants_util
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(quadrants_util PUBLIC quadrants_common)
