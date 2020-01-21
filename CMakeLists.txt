cmake_minimum_required(VERSION 3.7)

set(PROJECT_VERSION 0.0.1)

project(Quartz VERSION ${PROJECT_VERSION}
        DESCRIPTION "An simple adiabatic quantum dynamics library")

include(GNUInstallDirs)

enable_testing()

set(CMAKE_CXX_STANDARD 14)

add_library(quartz SHARED
            src/wrapper.cpp
        )

####################
#  find Libraries  #
####################
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

include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
include_directories(SYSTEM Catch)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(quartz ${ARMADILLO_LIBRARIES})

install(TARGETS quartz
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

####################
#       Test       #
####################


####################
#      Catch2      #
####################

add_library(catch2 SHARED include/Catch2/Catch_main.cpp)
target_include_directories(catch2 PUBLIC include/Catch2/)


add_executable(test_math
        test/details/math/polynomial_test.cpp
        )

target_link_libraries(test_math catch2 quartz)

add_test(NAME UNITTEST_math
         COMMAND bin/test_math)