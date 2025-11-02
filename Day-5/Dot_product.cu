#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

__global__ void dot_product(float* A, float* B, float* partialSums, int N)
{
	extern __shared__ float cache[]; //shared memory for partial_sums
	int globalId = blockDim.x * blockIdx.x + threadIdx.x; 
	int tid = threadIdx.x; 

	if(globalId<N) //so that only threads=N do the computation
	{
		cache[tid] = A[globalId] * B[globalId];
	}

	__syncthreads();

	for (int stride = blockDim.x / 2;stride > 0;stride /= 2)
	{
		if(tid<stride) //only half of the threads will work each time.
		cache[tid] +=cache[tid + stride];
		__syncthreads();
	}
	if (tid == 0) partialSums[blockIdx.x] = cache[0];

}

int main()
{
	int N = 10;
	size_t size = N * sizeof(float);
	float* A_h = new float[N];
	float* B_h = new float[N];

	for (int i = 0;i < N;i++) //for initialising the two vectors
	{
		A_h[i] = 1.0f;
		B_h[i] = 2.0f;
	}
		
	float* A_d, * B_d, * partial_d;
	int threadPerBlock = 16;
	int blocks = (N + threadPerBlock - 1) / threadPerBlock;
	cudaMalloc(&A_d, size);
	cudaMalloc(&B_d, size);
	cudaMalloc(&partial_d, blocks*sizeof(float));

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
	dot_product << <blocks, threadPerBlock,threadPerBlock* sizeof(float) >> > (A_d, B_d, partial_d, N);
	cudaDeviceSynchronize();

	float *partial_h = new float[blocks];
	cudaMemcpy(partial_h, partial_d, blocks * sizeof(float), cudaMemcpyDeviceToHost);

	float finalSum = 0;
	for (int i = 0;i<blocks;i++)
	{
		finalSum += partial_h[i];  //doing addiiton of the partial sum in the CPU.
	}
	cout << "Dot product: " << finalSum << endl;

	delete[]A_h;
	delete[]B_h;
	delete[]partial_h;
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(partial_d);
	return 0;

}

