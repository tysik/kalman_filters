cmake_minimum_required(VERSION 2.8)
project(KalmanFilters)

set(CMAKE_CXX_FLAGS "-std=gnu++11 -fpermissive ${CMAKE_CXX_FLAGS} -Wfatal-errors -Wno-write-strings")

find_package(Armadillo REQUIRED)
include_directories(/usr/include/python2.7)

add_executable(KF_test examples/kf_example.cpp kalman_filter.h)
target_link_libraries(KF_test ${ARMADILLO_LIBRARIES} libpython2.7.so)

add_executable(EKF_test examples/ekf_example.cpp extended_kalman_filter.h)
target_link_libraries(EKF_test ${ARMADILLO_LIBRARIES} libpython2.7.so)

add_executable(UKF_test examples/ukf_example.cpp unscented_kalman_filter.h)
target_link_libraries(UKF_test ${ARMADILLO_LIBRARIES} libpython2.7.so)
