#include<iostream>
#include<cstdio>
#include<cuda_runtime.h>
using namespace std;


// __device__ : Functie executata pe GPU, apelabila doar de pe GPU
// __host__ : Functie executata pe host, apelabila doar de pe host
// __global__ : Functie executata pe GPU, apelabila doar de pe host


__constant__ float c_gaussian[64];



void computeGaussianKernelCuda(float delta, int radius)
{
	float h_gaussian[64];
	for (int i = 0; i < 2 * radius + 1; ++i)
	{
		float x = i - radius;
		h_gaussian[i] = expf(-(x * x) / (2.0f * delta * delta));
	}
	cudaMemcpyToSymbol(c_gaussian, h_gaussian, sizeof(float)*(2 * radius + 1));
}

// Functie ce calculeaza distanta euclidiana dintre 2 puncte cu 4 coordonate
__device__  float euclideanLength(float3 a, float3 b, float d)
{
	float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z) ;
	return expf(-mod / (2.0f * d * d));
}


__device__  float3 multiplyCuda(float a, float3 b)
{
	return { a * b.x, a * b.y, a * b.z};
}


__device__  float3 addCuda(float3 a, float3 b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z};
}

__global__ void bilateralFilterKernel(float3*  deviceInput, float3*  deviceOutput, float euclideanDelta, int width, int height, int filterRadius)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // dimensiunea x a id-ului global 
	int idy = blockIdx.y * blockDim.y + threadIdx.y; // dimensiunea y a id-ului global 

	if ((idx < width) && (idy < height))
	{
		float sum = 0.0f;
		float3 t = { 0.f, 0.f, 0.f };
		int position = idy * width + idx;
		float3 center = deviceInput[position];
		int r = filterRadius;

		float domainDistance = 0.0f, colorDistance = 0.0f, factor = 0.0f;

		for (int i = -r; i <= r; ++i)
		{
			int currentY = idy + i;
			// verificam ca pozitiile y ale pixelilor invecinati sa nu depaseasca marginile matricei
			if (currentY < 0)
				currentY = 0;
			else if (currentY >= height)
				currentY = height - 1;

			for (int j = -r; j <= r; ++j)
			{
				// verificam ca pozitiile x ale pixelilor invecinati sa nu depaseasca marginile matricei
				int currentX = idx + j;
				if (currentX < 0)
					currentX = 0;
				else if (currentX >= width)
					currentX = width - 1;

				float3 currentPixel = deviceInput[currentY * width + currentX];
				domainDistance = c_gaussian[r + i] * c_gaussian[r + j];
				colorDistance = euclideanLength(currentPixel, center, euclideanDelta);
				factor = domainDistance * colorDistance;
				sum += factor;
				t = addCuda(t, multiplyCuda(factor, currentPixel));
			}
		}

		deviceOutput[position] = multiplyCuda(1.f / sum, t);
	}
}



void bilateralFilterCuda(float3*  hostInput, float3*  hostOutput, float euclideanDelta, int width, int height, int filterRadius)
{
	// compute the gaussian kernel for the current radius and delta
	computeGaussianKernelCuda(euclideanDelta, filterRadius);

	int inputBytes = width * height * sizeof(float3);
	int outputBytes = inputBytes;

	float3* deviceInput, *deviceOutput;
	cudaMalloc<float3>(&deviceInput, inputBytes);
	cudaMalloc<float3>(&deviceOutput, outputBytes);


	cudaMemcpy(deviceInput, hostInput, inputBytes, cudaMemcpyHostToDevice); // copiem datele in memoria GPU


	dim3 block(8, 8); //definim un bloc de 8x8 threaduri
	dim3 grid((width + block.x - 1) , (height + block.y - 1) );// definim gridul astfel incat sa acopere toata imaginea


	bilateralFilterKernel << <grid, block >> > (deviceInput, deviceOutput, euclideanDelta, width, height, filterRadius);


	cudaDeviceSynchronize(); // blocheaza CPU pana toate apelurile CUDA se finalizeaza

	cudaMemcpy(hostOutput, deviceOutput, outputBytes, cudaMemcpyDeviceToHost); // copiem rezultatul de pe GPU pe CPU

	cudaFree(deviceInput);
	cudaFree(deviceOutput);
}