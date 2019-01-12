#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector> // for vector
#include <iterator> // for std::istream_iterator and std::ostream_iterator
#include <algorithm> // for std::copy
using namespace std;
using namespace cv;



void readImageOpenCV() {
	Mat img = imread("flower.jpg");
	namedWindow("image", WINDOW_NORMAL);
	imshow("image", img);
	waitKey(0);
}


int main() {

	// Define file stream object, and open the file
	ifstream inputFile("flower.jpg", ios::binary);
	ofstream outputFile("flower1.jpg", ios::binary);


	// Reading the file content using the iterator!
	
	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(inputFile), {});

	
	copy(buffer.begin(), buffer.end(), ostreambuf_iterator<char>(outputFile));


	cout << "Done";
	return 0;
}


