#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

__global__ void dotProduct(float* A, float* B, float* partialsums, int N)
{
	extern __shared__ float cache[];
	int globalId = blockDim.x + blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	if (globalId < N)
		cache[tid] = A[globalId] * B[globalId];
	__syncthreads();

	for (int stride = blockDim.x / 2;stride > 0;stride /= 2)
	{
		if (tid < stride)
			cache[tid] += cache[tid + stride];
		__syncthreads();
	}
	if (tid == 0) partialsums[blockIdx.x] = cache[0];
}
__global__ void finalReduction(float* partialsums, float *output, int N)
{
	extern __shared__ float cache[];
	int globalId = blockDim.x + blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	if (globalId < N)
		cache[tid] = partialsums[globalId];
	else
		cache[tid] = 0.0f;
	for (int stride = blockDim.x / 2;stride > 0;stride /= 2)
	{
		if (tid < stride)
			cache[tid] += cache[tid + stride];
		__syncthreads();
	}
	if (tid == 0)output[blockIdx.x] = cache[0];
}

int main()
{
	int N = 1 << 20;
	size_t size = N*sizeof(float);
	int threads = 512;
	int blocks = (N + threads - 1) / threads;
	//(mistake) float* A_h, * B_h, * partial_h;
	float* A_h = new float[N];
	float* B_h = new float[N];
	//float* partial_h = new float[N];

	float* A_d, * B_d, * partial_d;

	for (int i = 0;i < N;i++)
	{
		A_h[i] = 1.0f;
		B_h[i] = 2.0f;
	}

	cudaMalloc(&A_d, size);
	cudaMalloc(&B_d, size);
	cudaMalloc(&partial_d, blocks*sizeof(float));

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	dotProduct << <blocks, threads, blocks * sizeof(float) >> > (A_d,B_d,partial_d,N);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Kernel execution time: " << elapsedTime << " ms" << endl;

	float* input = partial_d;
	float* output = nullptr;
	int newN = blocks;
	
	while(newN>1)
	{
		int newthreads = 512;
		int newblocks = (newN + threads - 1) / threads;
		cudaMalloc(&output, newblocks * sizeof(float));
		finalReduction << <newblocks, newthreads, newblocks * sizeof(float) >> > (input, output, newN);
		cudaDeviceSynchronize();
		cudaFree(input);
		input = output;
		newN = newblocks;
	}
	float result;
	cudaMemcpy(&result, input, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(input);
	delete[]A_h;
	delete[]B_h;
	cudaFree(A_d);
	cudaFree(B_d);
	return 0;
	

}
