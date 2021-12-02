# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamplesCapture)

find_package(OpenCV REQUIRED)

if(BUILD_GRAPHICAL_EXAMPLES)
    add_executable(rs-capture rs-capture.cpp ../example.hpp)
    set_property(TARGET rs-capture PROPERTY CXX_STANDARD 11)
    target_link_libraries(rs-capture ${DEPENDENCIES} ${OpenCV_LIBS}
                            /home/zh/code/darknet/libdarknet.so          
                            /usr/local/cuda-10.0/lib64/libcudart.so.10.0 
                            /usr/local/cuda-10.0/lib64/libcudnn.so.7
                            /usr/local/cuda-10.0/lib64/libcurand.so.10.0 
                            /usr/local/cuda-10.0/lib64/libcublas.so.10.0
    )
    include_directories(../ ../../third-party/tclap/include ../../examples 
                        ${openCV_INCLUDE_DIRS}
  )
    link_directories(${OpenCV_LIBRARY_DIRS})
    add_definitions(${OpenCV_DEFINITIONS})
    set_target_properties (rs-capture PROPERTIES FOLDER "Examples")

    install(TARGETS rs-capture RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()