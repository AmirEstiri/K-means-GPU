#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
using namespace std;
#define n 100
#define K 10
#define d 3

//////////////////////////////////////////
__global__ void assignCluster(int* points)
{
    int tid = threadId.x;
}


int main(void)
{
    float *points;
    float *d_points;
    int size = n * d * sizeof(float);
    int i;

    points = (float*)malloc(size); 
    cudaMalloc((void*) &d_points, size);
    
    for(i=0 ; i<n*d ; i++)
    	points[i] = (rand() % 10);

    cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(n);
    dim3 dimGrid(1);
    assignCluster <<< dimGrid, dimBlock >>> (d_points);
    cudaMemcpy( out, d_out, size, cudaMemcpyDeviceToHost );
	return 0;
}
