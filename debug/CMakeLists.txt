# Version Debug

############# Test for features.cpp #############
# PROJECT(test_features)
# CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# FIND_PACKAGE(OpenCV REQUIRED)
# INCLUDE_DIRECTORIES(CMAKE_CURRENT_SOURCE_DIR/include)
# ADD_EXECUTABLE(test_features_exe test_features.cpp features.cpp)
# TARGET_LINK_LIBRARIES(test_features_exe ${OpenCV_LIBS})

############# Test for svm_tools.cpp #############
PROJECT(test_svm_tools)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(CMAKE_CURRENT_SOURCE_DIR/include)
ADD_EXECUTABLE(test_svm_tools_exe svm_tools.cpp test_svm_tools.cpp features.cpp)
TARGET_LINK_LIBRARIES(test_svm_tools_exe ${OpenCV_LIBS})