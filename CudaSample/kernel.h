#include<cuda_runtime.h>

void bilateralFilterCuda(float4*  hostInput, float4*  hostOutput, float euclideanDelta, int width, int height, int filterRadius);
