cmake_minimum_required(VERSION 3.10)
project(BlackjackJoc)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(main_app main.cpp)

target_link_libraries(main_app PRIVATE ${OpenCV_LIBS})