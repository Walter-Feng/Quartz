cmake_minimum_required(VERSION 3.7)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(PROJECT_VERSION 0.0.1)

project(Quartz VERSION ${PROJECT_VERSION}
        DESCRIPTION "An simple adiabatic quantum dynamics library")

IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    FIND_PACKAGE(Git)
    IF(GIT_FOUND)
        EXECUTE_PROCESS(
                COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE "QUARTZ_BUILD_VERSION"
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        MESSAGE( STATUS "Git version: ${QUARTZ_BUILD_VERSION}" )
    ELSE(GIT_FOUND)
        SET(QUARTZ_BUILD_VERSION 0)
    ENDIF(GIT_FOUND)
ENDIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/include/quartz_internal/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/quartz_internal/version.h)

include(GNUInstallDirs)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/quartz_internal)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/quartz_internal)

enable_testing()

set(CMAKE_CXX_STANDARD 17)

add_library(quartz SHARED
        src/wrapper.cpp
        )

####################
#  find Libraries  #
####################

#####Armadillo######

find_package(
        Armadillo
        8.500.0
        REQUIRED
)

if (${ARMADILLO_FOUND})
    message("Found Armadillo Library,")
    message("     ARMADILLO_VERSION: " ${ARMADILLO_VERSION_STRINcatchG})
    message("     ARMADILLO_LIBRARIES   : " ${ARMADILLO_LIBRARIES})
    message("     ARMADILLO_INCLUDE_DIRS: " ${ARMADILLO_INCLUDE_DIRS})

    include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(quartz ${ARMADILLO_LIBRARIES})
endif ()

find_package(
        HDF5
)

if (${HDF5_FOUND})
    message("Found HDF5 Library,")
    message("     HDF5_VERSION: " ${HDF5_VERSION_STRINcatchG})
    message("     HDF5_LIBRARIES   : " ${HDF5_LIBRARIES})
    message("     HDF5_INCLUDE_DIRS: " ${HDF5_INCLUDE_DIRS})

    include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
    target_link_libraries(quartz ${HDF5_LIBRARIES})
endif ()
######OpenMP######

FIND_PACKAGE(OpenMP)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

install(TARGETS quartz
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h*")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "quartz")

######GSL######

find_package(
        GSL
        2.4
        REQUIRED
)

include_directories(SYSTEM ${GSL_INCLUDE_DIRS})
target_link_libraries(quartz ${GSL_LIBRARIES})



################################
#  Compile external libraries  #
################################


#######Catch2#######

add_library(catch2 SHARED external/Catch2/Catch_main.cpp)
include_directories("external/Catch2")
target_include_directories(catch2 PUBLIC external/Catch2/)

########fmt#########

message("\n This is {fmt} library.")
add_subdirectory(external/fmt)
include_directories("external/fmt/include")
target_link_libraries(quartz fmt::fmt-header-only)
add_definitions(-DHAVE_FMT)
message("\n")

#######################
# Compile Executables #
#######################

option(HAVE_EXE "Have Executables" ON)

if (HAVE_EXE)

    option(ARGS_BUILD_EXAMPLE OFF)
    option(ARGS_BUILD_UNITTESTS OFF)
    ########args#########
    add_subdirectory(external/args)
    include_directories("external/args")
    target_link_libraries(quartz args)
    add_definitions(-DHAVE_ARGS)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    ######Boost######

    find_package(
            Boost
            1.45.0
    )

    if (${Boost_FOUND})
        add_executable(quartz.exe
                src/main.cpp
                src/run.cpp
                src/file_system/json_parse.cpp
                src/util/ptree.cpp
                src/util/resolve.cpp
                src/parse/math/exponential.cpp
                src/parse/math/polynomial.cpp
                src/parse/math/gaussian.cpp
                src/parse/math/math_object.cpp
                src/parse/methods/m_dvr.cpp
                )

        include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
        include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})

        target_link_libraries(quartz ${Boost_LIBRARIES})

    else ()
        add_executable(quartz.exe
                src/main_without_boost.cpp
                )
    endif ()

    target_link_libraries(quartz.exe quartz)

endif ()

####################
#       Test       #
####################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

add_executable(test_core
        test/printer_test.cpp
        test/propagate_test.cpp
        )

target_link_libraries(test_core catch2 quartz)

add_test(NAME UNITTEST_core
        COMMAND bin/test_math)

add_executable(test_mathematics
        test/details/math/exponential_test.cpp
        test/details/math/polynomial_test.cpp
        test/details/math/gaussian_test.cpp
        test/details/math/runge_kutta_test.cpp
        test/details/math/schrotinger_wrapper_test.cpp
        test/details/math/moyal_bracket_test.cpp
        test/details/math/taylor_expand_test.cpp
        test/details/math/math_object_test.cpp
        )

target_link_libraries(test_mathematics catch2 quartz)

add_test(NAME UNITTEST_math
        COMMAND test/test_math)

add_executable(test_utils
        test/utils/check_member_test.cpp
        test/utils/elementary_function_operator.cpp
        test/utils/gsl_converter_test.cpp)

target_link_libraries(test_utils catch2 quartz)

add_test(NAME UNITTEST_utils
        COMMAND test/test_utils)

add_executable(test_methods
        test/details/methods/dvr_test.cpp
        test/details/methods/md_test.cpp
        test/details/methods/fixed_gaussian_basis_test.cpp
        test/details/methods/wd_test.cpp
        test/details/methods/dmd_test.cpp
        test/details/methods/cwa_smd_test.cpp
        test/details/methods/cwa_smd_opt_test.cpp
        test/details/methods/dvr_smd_test.cpp
        test/details/methods/g_cwa_smd_test.cpp
        #test/details/methods/heller_cwa_test.cpp
        #test/details/methods/heller_gaussian_test.cpp
        )

target_link_libraries(test_methods catch2 quartz)

add_test(NAME UNITTEST_methods
        COMMAND test/test_methods)