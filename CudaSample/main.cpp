#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#include "kernel.h";

using namespace std;
using namespace cv;


int main(int argc, char **argv) {

	cv::Mat input = cv::imread("flower.jpg", IMREAD_UNCHANGED);


	///convert from char(0-255) BGR to float (0.0-0.1) RGBA
	Mat inputRGB;
	cvtColor(input, inputRGB, 4, 3); // convert BGR to RGB
	inputRGB.convertTo(inputRGB, CV_32F); // CV_32F -> valoare pixeli 0-1
	inputRGB /= 255;

	Mat output(input.size(), inputRGB.type());

	float euclideanDelta = 3.0f;
	int filterRadius = 3;


	bilateralFilterCuda((float3*)inputRGB.ptr<float3>(), (float3*)output.ptr<float3>(), euclideanDelta, inputRGB.cols, inputRGB.rows, filterRadius);
	// convert back to char (0-255) BGR
	output *= 255;
	output.convertTo(output, CV_8U); //  CV_8U -> valoare pixeli 0-255
	cvtColor(output, output, cv::COLOR_BGR2RGB, 3);
	imwrite("output.jpg", output);
	return 0;
}
