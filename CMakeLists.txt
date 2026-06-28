cmake_minimum_required(VERSION 3.18)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# C++20 required for std::span (used by the BVH). CUDA 13.x Thrust supports C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
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
    src/core/main.h
    src/core/utilities.h
    src/core/cudaUtil.h
    src/scene/scene.h
    src/scene/sceneStructs.h
    src/scene/meshLoader.h
    src/scene/bvh.h
    src/render/pathtrace.h
    src/render/deviceScene.h
    src/render/intersections.h
    src/render/interactions.h
    src/render/bsdf.h
    src/render/warp.h
    src/display/preview.h
    src/display/glslUtility.hpp
    src/display/image.h
)

set(sources
    src/core/main.cpp
    src/core/utilities.cpp
    src/scene/scene.cpp
    src/scene/meshLoader.cpp
    src/scene/bvh.cpp
    src/render/pathtrace.cu
    src/render/deviceScene.cu
    src/render/intersections.cu
    src/render/interactions.cu
    src/render/warp.cu
    src/render/bsdf.cu
    src/display/preview.cpp
    src/display/glslUtility.cpp
    src/display/image.cpp
    src/display/stb.cpp
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

# ----------------------------------------------------------------------------
# Intel Open Image Denoise (OIDN) — built from the libs/oidn submodule.
# CUDA-only device build: this avoids the ISPC compiler and TBB, which are only
# required by OIDN's CPU device. Needs only the CUDA toolkit + CUTLASS (already
# pulled in as a nested submodule under libs/oidn/external/cutlass).
# Note: the CUTLASS kernels make the first build significantly slower.
# ----------------------------------------------------------------------------
option(USE_OIDN "Enable Intel Open Image Denoise" ON)
if(USE_OIDN)
    set(OIDN_DEVICE_CPU  OFF CACHE BOOL "" FORCE)
    set(OIDN_DEVICE_CUDA ON  CACHE BOOL "" FORCE)
    set(OIDN_APPS        OFF CACHE BOOL "" FORCE)
    add_subdirectory(libs/oidn)
endif()

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
# src subfolders so the project's flat #include "foo.h" resolve regardless of
# which subfolder a file now lives in.
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/scene
    ${CMAKE_SOURCE_DIR}/src/render
    ${CMAKE_SOURCE_DIR}/src/display
)
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    cudadevrt
    stream_compaction  # TODO: uncomment if using your stream compaction
    )
if(USE_OIDN)
    # OpenImageDenoise carries its own INTERFACE include dirs, so
    # #include <OpenImageDenoise/oidn.hpp> works without extra paths.
    target_link_libraries(${CMAKE_PROJECT_NAME} OpenImageDenoise)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE USE_OIDN)

    # OIDN emits its DLLs into its own binary dir (build/libs/oidn), not next to
    # the exe. The CUDA device module is built as an ExternalProject and is loaded
    # dynamically at runtime (not a link dependency), so force it to build and copy
    # all OIDN runtime DLLs next to the path tracer so they're found at launch.
    add_dependencies(${CMAKE_PROJECT_NAME} OpenImageDenoise_device_cuda)
    if(WIN32)
        add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -DOIDN_BIN_DIR=${CMAKE_BINARY_DIR}/libs/oidn
                -DDST_DIR=$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>
                -P ${CMAKE_SOURCE_DIR}/cmake/copy_oidn_dlls.cmake
            COMMENT "Copying OIDN runtime DLLs next to the path tracer")
    endif()
endif()
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
