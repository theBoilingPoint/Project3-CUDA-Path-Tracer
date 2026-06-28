# Copy OIDN runtime DLLs next to the path tracer executable.
#
# Run as: cmake -DOIDN_BIN_DIR=<oidn build dir> -DDST_DIR=<exe dir> -P copy_oidn_dlls.cmake
#
# The main library (OpenImageDenoise.dll, OpenImageDenoise_core.dll) is a normal
# CMake target, but the CUDA device module (OpenImageDenoise_device_cuda.dll) is
# built by an ExternalProject and lands in an install-prefixed subdirectory whose
# exact path depends on the generator/config. Rather than hardcode that, we glob
# the whole OIDN build tree for the runtime DLLs and copy whatever we find.
file(GLOB_RECURSE _oidn_dlls "${OIDN_BIN_DIR}/*OpenImageDenoise*.dll")
foreach(_dll ${_oidn_dlls})
    file(COPY "${_dll}" DESTINATION "${DST_DIR}")
endforeach()
