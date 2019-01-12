#include<cuda_runtime.h>

void bilateralFilterCuda(float3*  hostInput, float3*  hostOutput, float euclideanDelta, int width, int height, int filterRadius);
