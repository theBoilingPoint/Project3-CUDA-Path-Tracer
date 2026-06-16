@echo off
setlocal

set ROOT=%~dp0..
set BUILD_DIR=%ROOT%\build
set NINJA_DIR=%ROOT%\build_ninja

:: --- Main build (Visual Studio) ---
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G "Visual Studio 18 2026"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build "%BUILD_DIR%" --config Debug
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: --- Ninja build for compile_commands.json ---
:: Initialize VS developer environment so nvcc can find cl.exe
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake -S "%ROOT%" -B "%NINJA_DIR%" -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: --- Copy compile_commands.json and clean up ---
copy /Y "%NINJA_DIR%\compile_commands.json" "%BUILD_DIR%\compile_commands.json"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rmdir /S /Q "%NINJA_DIR%"

endlocal
