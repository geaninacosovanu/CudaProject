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


	cv::Mat input = cv::imread("castle.jpg", IMREAD_UNCHANGED);


	///convert from char(0-255) BGR to float (0.0-0.1) RGBA
	Mat inputRGBA;
	cvtColor(input, inputRGBA, 2, 4);
	inputRGBA.convertTo(inputRGBA, CV_32FC4);
	inputRGBA /= 255;

	//Create output image
	Mat output(input.size(), inputRGBA.type());
	//Mat output(input.size(), input.type());

	const float euclidean_delta = 3.0f;
	const int filter_radius = 3;
	/*
		bilateralFilterCuda((float4*)input.ptr<float4>(),
			(float4*)output.ptr<float4>(),
			euclidean_delta,
			input.cols, input.rows,
			filter_radius);*/

	bilateralFilterCuda((float4*)inputRGBA.ptr<float4>(),
		(float4*)output.ptr<float4>(),
		euclidean_delta,
		inputRGBA.cols, inputRGBA.rows,
		filter_radius);
	// convert back to char (0-255) BGR
	output *= 255;
	//output.convertTo(output, CV_8UC4);
	cvtColor(output, output, 2, 3);
	imwrite("output.jpg", output);
	return 0;
}
