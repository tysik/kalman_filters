cmake_minimum_required(VERSION 2.8)
project(kalman_filters)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)
include_directories(/usr/include/python2.7)

add_executable(KF_example examples/kf_example.cpp kalman_filter.h)
target_link_libraries(KF_example ${ARMADILLO_LIBRARIES} libpython2.7.so)

add_executable(EKF_example examples/ekf_example.cpp extended_kalman_filter.h)
target_link_libraries(EKF_example ${ARMADILLO_LIBRARIES} libpython2.7.so)

add_executable(UKF_example examples/ukf_example.cpp unscented_kalman_filter.h)
target_link_libraries(UKF_example ${ARMADILLO_LIBRARIES} libpython2.7.so)
