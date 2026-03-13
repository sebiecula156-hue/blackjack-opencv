cmake_minimum_required(VERSION 3.10)
project(BlackjackJoc)

# Această linie caută OpenCV pe calculatorul tău
find_package(OpenCV REQUIRED)

# Afișează un mesaj în terminal dacă a găsit OpenCV (ca să fim siguri)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(main_app main.cpp)

# Leagă biblioteca de programul tău
target_link_libraries(main_app PRIVATE ${OpenCV_LIBS})