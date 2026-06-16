cmake_minimum_required(VERSION 3.18)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CUDA 13.x Thrust requires C++17
set(CMAKE_CXX_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(PREBUILT "${CMAKE_SOURCE_DIR}/libs/prebuilt")

    set(GLFW_ROOT_DIR ${PREBUILT})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${PREBUILT})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

include_directories(${CMAKE_SOURCE_DIR}/libs/glm)
include_directories(${CMAKE_SOURCE_DIR}/libs/json/single_include)
include_directories(${CMAKE_SOURCE_DIR}/libs/stb)

# CUDA 12+ moved Thrust/CUB into include/cccl/; derive root from nvcc path
get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_ROOT "${CUDA_BIN_DIR}" DIRECTORY)
if(EXISTS "${CUDA_ROOT}/include/cccl")
    include_directories("${CUDA_ROOT}/include/cccl")
endif()

set(headers
    src/main.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/preview.h
    src/utilities.h
    src/warp.h
    src/bsdf.h
    src/meshLoader.h
    src/bvh.h
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/intersections.cu
    src/interactions.cu
    src/scene.cpp
    src/preview.cpp
    src/utilities.cpp
    src/warp.cu
    src/bsdf.cu
    src/meshLoader.cpp
    src/bvh.cpp
)

set(imgui_headers
    libs/ImGui/imconfig.h
    libs/ImGui/imgui.h
    libs/ImGui/imgui_internal.h
    libs/ImGui/imstb_rectpack.h
    libs/ImGui/imstb_textedit.h
    libs/ImGui/imstb_truetype.h
    libs/ImGui/backends/imgui_impl_glfw.h
    libs/ImGui/backends/imgui_impl_opengl3.h
    libs/ImGui/backends/imgui_impl_opengl3_loader.h
)

set(imgui_sources
    libs/ImGui/imgui.cpp
    libs/ImGui/imgui_demo.cpp
    libs/ImGui/imgui_draw.cpp
    libs/ImGui/imgui_tables.cpp
    libs/ImGui/imgui_widgets.cpp
    libs/ImGui/backends/imgui_impl_glfw.cpp
    libs/ImGui/backends/imgui_impl_opengl3.cpp
)

set(tinyobj_headers
    libs/tinyobjloader/tiny_obj_loader.h
)

set(tinyobj_sources
    libs/tinyobjloader/tiny_obj_loader.cc
)

set(tinygltf_headers
    libs/tinygltf/tiny_gltf.h
)

set(tinygltf_sources
    libs/tinygltf/tiny_gltf.cc
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)
list(SORT tinyobj_headers)
list(SORT tinyobj_sources)
list(SORT tinygltf_headers)
list(SORT tinygltf_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})
source_group("tinyobj\\Headers" FILES ${tinyobj_headers})
source_group("tinyobj\\Sources" FILES ${tinyobj_sources})
source_group("tinygltf\\Headers" FILES ${tinygltf_headers})
source_group("tinygltf\\Sources" FILES ${tinygltf_sources})

#add_subdirectory(src/ImGui)
add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/libs ${CMAKE_SOURCE_DIR}/libs/ImGui)
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    cudadevrt
    stream_compaction  # TODO: uncomment if using your stream compaction
    )
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
if(MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>")
    # C++17 introduces std::byte; suppress ambiguity with Windows SDK's byte typedef
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _HAS_STD_BYTE=0)
    # Prevent Windows SDK from defining min/max macros that break std:: and glm:: calls
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE NOMINMAX)
endif()
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})
