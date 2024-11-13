#ifndef YOLOV10_HPP
#define YOLOV10_HPP

#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
  
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

struct DetResult {
    cv::Rect bbox; ///< Bounding box of the detected object.
    float conf;    ///< Confidence score of the detection.
    int label;     ///< Label of the detected object.

    
    DetResult(cv::Rect bbox, float conf, int label) : bbox(bbox), conf(conf), label(label) {}
};

class YOLOv10 {
public:
   
    YOLOv10();

    ~YOLOv10();

   
    void preProcess(cv::Mat* img, int length, float* factor, std::vector<float>& data);

   
    std::vector<DetResult> postProcess(float* result, float factor, int outputLength);

    
    void drawBbox(cv::Mat& img, std::vector<DetResult>& res);

   
    std::shared_ptr<nvinfer1::IExecutionContext> createExecutionContext(const std::string& modelPath);

    void inferVideo(const std::string& videoPath, const std::string& enginePath);

    
    void inferImage(const std::string& imagePath, const std::string& enginePath);

   
    void convertOnnxToEngine(const std::string& onnxFile, int memorySize);

private:
    Logger logger; ///< Logger instance for TensorRT.
};

#endif // YOLOV10_HPP
