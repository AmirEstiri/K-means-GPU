#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
using namespace std;

#define BLOCK_SIZE 1024
#define n 100
#define K 10
#define d 3

struct Point{
float val[d];
};

__device__ float distance(Point* p1, Point* p2){
    float dist = 0;
    float tmp;
    for (int i = 0 ; i < d; i ++){
            tmp = (p1->val[i] - p2->val[i]);
            dist += tmp * tmp;
    }
    return dist;
}
__global__ void assignCluster(Point *data, int *cluster, Point *centers)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > n)
            return;

    int best_cluster = 0;
    float best_distance = 1e10;
    float tmp;

    for (int i = 0 ; i < K; i++){
            tmp = distance(data + i, centers + i);
            if (tmp > best_distance) {
                    best_distance = tmp;
                    best_cluster = i;
            }
    }
cluster[tid] = best_cluster;

}


__global__ void updateCenters(Point* data, int* cluster, Point* centers, int* cluster_sizes)
{

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int id = threadIdx.x;

    __shared__ Point data_shared[BLOCK_SIZE];
    __shared__ int cluster_shared[BLOCK_SIZE];
    __shared__ int cluster_sizes_shared[K];
    __shared__ Point cluster_sum[BLOCK_SIZE];

    if (id < K)
        cluster_sizes_shared[id] = 0;

    cluster_shared[id] = cluster[tid];
    data_shared[id]= data[tid];

    if (tid < K){
      for (int i = 0 ; i < d; i++)
          centers[tid].val[i] = 0;
        cluster_sizes[tid] = 0;
    }
__syncthreads();

for (int c = 0 ; c < K ; c++){
    if (cluster_shared[id] == c){
        for (int i = 0 ; i < d; i++)
            cluster_sum[id].val[i] = data_shared[id].val[i];
        atomicAdd(cluster_sizes_shared + c, 1);
    } else {
        for (int i = 0 ; i < d; i++)
            cluster_sum[id].val[i] = 0;
    }

    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (id < s)
            for (int ind = 0; ind < d; ind++)
        cluster_sum[id].val[ind] += cluster_sum[id+s].val[ind];
        __syncthreads();
    }

    if (id == 0){
        for (int ind = 0; ind < d; ind++)
            centers[c].val[ind] += cluster_sum[0].val[ind];
        atomicAdd(cluster_sizes + c, cluster_sizes_shared[c]);
    }
}
__syncthreads();
    if(tid < K){
        for (int ind = 0; ind < d; ind++)
        centers[tid].val[ind] = centers[tid].val[ind]/cluster_sizes[tid];
    }

}


int main(void)
{
    Point *data;
    int *cluster;
    Point *centers;

    Point *d_data;
    int *d_cluster;
    Point *d_centers;

    int size_data = n * sizeof(Point);
    int size_cluster = n * sizeof(int);
    int size_centers = K * sizeof(Point);

    data = (Point*)malloc(size_data);
    cluster = (int*)malloc(size_cluster);
    centers = (Point*)malloc(size_centers);

    cudaMalloc((void**)&d_data, size_data);
    cudaMalloc((void**)&d_cluster, size_cluster);
cudaMalloc((void**)&d_centers, size_centers);

int i, j;
for(i=0 ; i<n ; i++){
    for (j=0; j<d; j++)
        (data[i].val)[j] = i;
}

for(i=0 ; i<K ; i++){
    for (j=0; j<d; j++)
        (centers[i].val)[j] = (rand() % 10);
}

for (i=0; i<n; i++)
    cluster[i] = 0;

cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice);
cudaMemcpy(d_centers, centers, size_centers, cudaMemcpyHostToDevice);

dim3 dimBlock(n);
dim3 dimGrid(1);

// printf("%f\n", distance(data, data + 1));

assignCluster <<< dimGrid, dimBlock >>> (d_data, d_cluster, d_centers);
cudaMemcpy( cluster, d_cluster, size_cluster, cudaMemcpyDeviceToHost );
    return 0;
}