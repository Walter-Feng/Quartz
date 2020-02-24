cmake_minimum_required(VERSION 3.7)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(PROJECT_VERSION 0.0.1)

project(Quartz VERSION ${PROJECT_VERSION}
        DESCRIPTION "An simple adiabatic quantum dynamics library")

include(GNUInstallDirs)

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

if( ${ARMADILLO_FOUND} )
    message("Found Armadillo Library,")
    message("     ARMADILLO_VERSION: " ${ARMADILLO_VERSION_STRINcatchG})
    message("     ARMADILLO_LIBRARIES   : " ${ARMADILLO_LIBRARIES})
    message("     ARMADILLO_INCLUDE_DIRS: " ${ARMADILLO_INCLUDE_DIRS})
endif()

######OpenMP######

FIND_PACKAGE(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/quartz_internal)
target_link_libraries(quartz ${ARMADILLO_LIBRARIES})

install(TARGETS quartz
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h*")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "quartz")

########fmt#########

add_subdirectory(external/fmt)
include_directories ("external/fmt/include")
target_include_directories(fmt PUBLIC external/fmt/include)
target_link_libraries(quartz fmt)
add_definitions(-DHAVE_FMT)

####################
#       Test       #
####################


####################
#      Catch2      #
####################

add_library(catch2 SHARED external/Catch2/Catch_main.cpp)
include_directories("external/Catch2")
target_include_directories(catch2 PUBLIC external/Catch2/)

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
        )

target_link_libraries(test_mathematics catch2 quartz)

add_executable(test_utils
        test/utils/check_member_test.cpp
        test/utils/elementary_function_operator.cpp)

target_link_libraries(test_utils catch2 quartz)

add_executable(test_methods
        test/details/methods/dvr_test.cpp
        test/details/methods/md_test.cpp
        )

target_link_libraries(test_methods catch2 quartz)

add_test(NAME UNITTEST_math
         COMMAND bin/test_math)