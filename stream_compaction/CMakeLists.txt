set(headers
    "thrust.h"
    )

set(sources
    "thrust.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(stream_compaction ${sources} ${headers})

get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_ROOT "${CUDA_BIN_DIR}" DIRECTORY)
if(EXISTS "${CUDA_ROOT}/include/cccl")
    target_include_directories(stream_compaction PRIVATE "${CUDA_ROOT}/include/cccl")
endif()

if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
if(MSVC)
    target_compile_options(stream_compaction PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()