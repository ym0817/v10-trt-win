#include "inference.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
	//const std::string weightPath = "E:\\v10\\ADC\\ADC\\best.onnx";
	//bool isgpu(true);
	////bool isgpu(false);
	//std::string namefile = "E:\\v10\\ADC\\ADC\\class.names";
	//int	threadNum = 1;
	//bool init = LoadADCModel_Main(weightPath, isgpu, namefile, threadNum);


	//std::string imgPath = "E:\\v10\\ADC\\ADC\\192_r.JPG";
	/*std::string imgPath = "E:\\v10\\ADC\\ADC\\185-FAB-41.JPG";
	float confidence_threshold = 0.25;
	std::vector<int> classIDVec;
	std::vector<float> scoreVec;

	for (int n =0 ; n < 10; n++ )
		bool ret = ADCModelInferenceImage_Main(imgPath, confidence_threshold, classIDVec, scoreVec);*/

	std::string  enginePath = "best.engine";
	std::string inputPath = "192_r.JPG";
	YOLOv10 yolov10;
	for (int n = 0; n < 10; n++)
	{
		
		auto start = std::chrono::steady_clock::now();
		//yolov10.inferVideo(inputPath, enginePath);  // Perform inference on video
		yolov10.inferImage(inputPath, enginePath);  // Perform inference on image
		auto end = std::chrono::steady_clock::now();
		//std::chrono::duration<double> spent = end - start;
		double spent_ms = std::chrono::duration<double, std::milli>(end - start).count();
		std::cout << "---------PerImage cost time:    " << spent_ms << "   ms " << std::endl;
	
	}


		
	return 0;



}


