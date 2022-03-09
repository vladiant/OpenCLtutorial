cmake_minimum_required(VERSION 3.10)

project(3_Array LANGUAGES CXX)

find_package(OpenCL REQUIRED)

configure_file(ProcessArray.cl . COPYONLY)

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        OpenCL::OpenCL
)
