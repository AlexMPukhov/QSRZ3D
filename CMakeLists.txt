cmake_minimum_required(VERSION 3.18)
project(QSRZ LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Kokkos: either an installed Kokkos (-DKokkos_ROOT=...) or fetched from GitHub.
# The backend (OpenMP, CUDA, HIP, SYCL, Serial) is chosen when Kokkos is built,
# e.g. -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON  for an A100.
option(QSRZ_FETCH_KOKKOS "Download and build Kokkos if not found" ON)
find_package(Kokkos 4.0 QUIET)
if(NOT Kokkos_FOUND)
  if(QSRZ_FETCH_KOKKOS)
    message(STATUS "Kokkos not found - fetching Kokkos 4.4.01")
    include(FetchContent)
    FetchContent_Declare(kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.4.01)
    FetchContent_MakeAvailable(kokkos)
  else()
    message(FATAL_ERROR "Kokkos not found; set Kokkos_ROOT or QSRZ_FETCH_KOKKOS=ON")
  endif()
endif()

add_library(qsrz_core
  src/Config.cpp
  src/RadialGrid.cpp
  src/Tridiag.cpp
  src/FieldSolver.cpp
  src/Profiles.cpp
  src/Parser.cpp
  src/PlasmaSpecies.cpp
  src/Beam.cpp
  src/Simulation.cpp
  src/Parallel.cpp
  src/OpenPMDWriter.cpp
  src/Laser.cpp
  src/Ionization.cpp)
target_include_directories(qsrz_core PUBLIC src)
target_link_libraries(qsrz_core PUBLIC Kokkos::kokkos)

# MPI: domain decomposition along xi with time pipelining (optional)
option(QSRZ_ENABLE_MPI "Build with MPI (decomposition along xi)" ON)
if(QSRZ_ENABLE_MPI)
  find_package(MPI COMPONENTS C CXX)
  if(MPI_CXX_FOUND)
    target_link_libraries(qsrz_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(qsrz_core PUBLIC QSRZ_USE_MPI)
    message(STATUS "QSRZ: MPI enabled (${MPI_CXX_LIBRARY_VERSION_STRING})")
  else()
    message(WARNING "QSRZ: MPI not found - building the serial version (set QSRZ_ENABLE_MPI=OFF to silence)")
  endif()
endif()

# openPMD output (optional): https://github.com/openPMD/openPMD-api, e.g. -DopenPMD_ROOT=/path/to/install
option(QSRZ_ENABLE_OPENPMD "Build with openPMD output (needs openPMD-api)" ON)
if(QSRZ_ENABLE_OPENPMD)
  find_package(openPMD 0.15 CONFIG QUIET)
  if(openPMD_FOUND)
    target_link_libraries(qsrz_core PUBLIC openPMD::openPMD)
    target_compile_definitions(qsrz_core PUBLIC QSRZ_USE_OPENPMD)
    message(STATUS "QSRZ: openPMD output enabled (openPMD-api ${openPMD_VERSION}, MPI: ${openPMD_HAVE_MPI})")
    if(MPI_CXX_FOUND AND NOT openPMD_HAVE_MPI)
      message(WARNING "QSRZ: openPMD-api has no MPI support: openPMD output only for single-rank runs")
    endif()
  else()
    message(STATUS "QSRZ: openPMD-api not found - native output only (set openPMD_ROOT to enable openPMD)")
  endif()
endif()

add_executable(qsrz src/main.cpp)
target_link_libraries(qsrz PRIVATE qsrz_core)

option(QSRZ_BUILD_TESTS "Build unit tests" ON)
if(QSRZ_BUILD_TESTS)
  enable_testing()
  add_executable(test_solvers tests/test_solvers.cpp)
  target_link_libraries(test_solvers PRIVATE qsrz_core)
  add_test(NAME solvers COMMAND test_solvers)
  add_executable(test_pushers tests/test_pushers.cpp)
  target_link_libraries(test_pushers PRIVATE qsrz_core)
  add_test(NAME pushers COMMAND test_pushers)
  add_executable(test_parser tests/test_parser.cpp)
  target_link_libraries(test_parser PRIVATE qsrz_core)
  add_test(NAME parser COMMAND test_parser)
  add_executable(test_profiles tests/test_profiles.cpp)
  target_link_libraries(test_profiles PRIVATE qsrz_core)
  add_test(NAME profiles COMMAND test_profiles)
endif()
