#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

#include <opencv2/opencv.hpp>

int main() {
    std::stringstream ss;

    ss << "nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=2 ! video/x-raw, width=480, height=680, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

    //ss << "nvarguscamerasrc !  video/x-raw(memory:NVMM), width=" << INPUT_WIDTH <<
    //", height=" << INPUT_HEIGHT <<
    //", format=NV12, framerate=" << CAMERA_FRAMERATE <<
    //" ! nvvidconv flip-method=" << FLIP <<
    //" ! video/x-raw, width=" << DISPLAY_WIDTH <<
    //", height=" << DISPLAY_HEIGHT <<
    //", format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

    cv::VideoCapture video;

    video.open(ss.str());

    if (!video.isOpened())
    {
        std::cout << "Unable to get video from the camera!" << std::endl;

        return -1;
    }

    std::cout << "Got here!" << std::endl;

    cv::Mat frame;

    video.read(frame);

    std::cout << "Finished!" << std::endl;
    cv::imwrite("test.jpg",frame);
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    return 0;
}
