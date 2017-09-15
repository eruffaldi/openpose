set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required( VERSION 2.8 )
project( StereoPose )

SET( CAFFE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/caffe)


find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS program_options REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( libuvc REQUIRED )
find_package( CUDA REQUIRED)
if (CUDA_FOUND)
    INCLUDE_DIRECTORIES(
        ${CUDA_INCLUDE_DIRS}
        ${CAFFE_DIR}/distribute/include
    )
    include_directories( include )
    include_directories( ${Boost_INCLUDE_DIRS} )
    include_directories( EIGEN3_INCLUDE_DIR )


    add_definitions(-DUSE_CAFFE)
    add_definitions(-DUSE_CUDNN)
    ADD_EXECUTABLE( stereopose src/stereocam.cpp src/stereopose.cpp src/utilities.cpp src/stereo_cam.cpp)
    TARGET_LINK_LIBRARIES(stereopose
        ${CUDA_LIBS}
        ${libuvc_LIBRARIES} 
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${CAFFE_DIR}/build/lib/libcaffe.so
        glog
        gflags
        openpose
    )
endif ()
