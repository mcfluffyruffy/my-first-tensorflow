#include <stdexcept>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
 
// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
 
// Colors.
cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0,0,255);


void DrawLabel(
        cv::Mat& inputImage,
        std::string label,
        int left,
        int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    cv::rectangle(inputImage, tlc, brc, BLACK);
    // Put the label on the black rectangle.
    putText(inputImage, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

std::vector<cv::Mat> PreProcess(
        cv::Mat &inputImage,
        cv::dnn::Net &net)
{
    // Convert to blob.
    cv::Mat blob;
    cv::dnn::blobFromImage(inputImage, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
 
    net.setInput(blob);
 
    // Forward propagate.
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
 
    return outputs;
}

cv::Mat PostProcess(
        cv::Mat inputImage,
        const std::vector<cv::Mat> &outputs,
        const std::vector<std::string> classList) 
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float xFactor = inputImage.cols / INPUT_WIDTH;
    float yFactor = inputImage.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classScores = data + 5;
            cv::Mat scores(1, classList.size(), CV_32FC1, classScores);
            cv::Point classId;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
            if (maxClassScore > SCORE_THRESHOLD)
            {
                confidences.push_back(confidence);
                classIds.push_back(classId.x);
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((cx - 0.5 * w) * xFactor);
                int top = int((cy - 0.5 * h) * yFactor);
                int width = int(w * xFactor);
                int height = int(h * yFactor);
                boxes.push_back(cv::Rect(
                    left,
                    top,
                    width,
                    height));
            }
        }
        data += 85;
    }

}

cv::Mat CaptureFrame()
{
    std::stringstream ss;
    ss << "nvarguscamerasrc sensor-id=0 !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=2 ! video/x-raw, width=480, height=680, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

    cv::VideoCapture video;

    video.open(ss.str());

    if (!video.isOpened())
    {
        throw std::runtime_error("Unable to get video from the camera!");
    }

    cv::Mat frame;

    video.read(frame);
    return frame;

}

cv::Mat CaptureDepthFrame()
{
    rs2::colorizer color_map;
    rs2::pipeline pipe;
    pipe.start();
    rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
    rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();
    cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
    return image;
}



std::vector<std::string> GetClassList(
        const std::string& classNameFilePath)
{
    std::vector<std::string> classList;
    std::ifstream stream(classNameFilePath);
    std::string line;
    while (getline(stream, line))
    {
        classList.push_back(line);
    }
    return classList;
}


int main() {
    cv::Mat frame = CaptureDepthFrame();
    //cv::dnn::Net net = cv::dnn::readNet("yolov4.onnx");
    //const std::vector<cv::Mat> detections = PreProcess(
    //        frame,
    //        net);
    //const std::vector<std::string> classList = GetClassList("coco.names"); 
//    cv::Mat detectionImage = PostProcess(
//            frame.clone(),
//            detections,
//            classList);
    std::cout << "Finished!" << std::endl;
    cv::imwrite("test.jpg",frame);
    return 0;
}
