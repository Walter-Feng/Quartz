cmake_minimum_required(VERSION 3.7)
project(Quartz VERSION 0.0.1
        DESCRIPTION "An simple adiabatic quantum dynamics library")

set(CMAKE_CXX_STANDARD 14)

add_library(libquartz SHARED
            src/math/polynomial.cpp
        )

####################
#  find armadillo  #
####################
find_package(
        Armadillo
        8.500.0
        REQUIRED
)

if( ${ARMADILLO_FOUND} )
    message("Found Armadillo Library,")
    message("     ARMADILLO_VERSION: " ${ARMADILLO_VERSION_STRING})
    message("     ARMADILLO_LIBRARIES   : " ${ARMADILLO_LIBRARIES})
    message("     ARMADILLO_INCLUDE_DIRS: " ${ARMADILLO_INCLUDE_DIRS})
endif()

include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(libquartz ${ARMADILLO_LIBRARIES})

