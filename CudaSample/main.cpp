/* *
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>

using namespace std;
using namespace cv;

void bilateralFilterCuda(const float4 * const h_input,
	float4 * const h_output,
	const float euclidean_delta,
	const int width, const int height,
	const int filter_radius);

int main(int argc, char **argv) {


	cv::Mat input = cv::imread("knowledge_graph_logo.jpeg", IMREAD_UNCHANGED);


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
