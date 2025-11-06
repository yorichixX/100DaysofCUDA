#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
using namespace std;

__global__ void vecAddGlobal(float* A, float* B, float* C, int n)
{
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalId < n)
		C[globalId] = A[globalId] + B[globalId];
}
__global__ void vecAddShared(float* A, float* B, float* C, int n)
{
	int globalId= blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	extern __shared__ float cache[];
	float* sA = cache;
	float* sB = cache + blockDim.x;

	if (globalId < n)
		{
			sA[tid] = A[globalId];
			sB[tid] = B[globalId];
		}
	__syncthreads();

	if (globalId < n)
		C[globalId] = sA[tid] + sB[tid];
}
int main()
{
	int n = 1 << 20;
	size_t size = n*sizeof(float);
	float* A_h = new float[n];
	float* B_h = new float[n];
	float* C_h = new float[n];

	for (int i = 0;i < n;i++)
	{
		A_h[i] = 1.0f;
		B_h[i] = 2.0f;
	}

	float* A_d, * B_d, * C_d;
	cudaMalloc(&A_d, size);
	cudaMalloc(&B_d, size);
	cudaMalloc(&C_d, size);
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	int threads = 512;
	int blocks = (n + threads - 1) / threads;

	vecAddGlobal << <1, 1 >> > (A_d, B_d, C_d, 1); //pre-warming the GPU!

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	vecAddGlobal << <blocks, threads >> > (A_d, B_d, C_d, n);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms1 = 0.0f;
	cudaEventElapsedTime(&ms1, start, stop);
	cout << "Global Vector addition time: " << ms1 << " ms" <<endl;

	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	cudaEvent_t a, b;
	cudaEventCreate(&a);
	cudaEventCreate(&b);
	cudaEventRecord(a);
	vecAddShared << <blocks, threads, 2 * threads * sizeof(float) >> > (A_d, B_d, C_d, n);
	cudaDeviceSynchronize();
	cudaEventRecord(b);
	cudaEventSynchronize(b);
	float ms2 = 0.0f;
	cudaEventElapsedTime(&ms2, a, b);
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
	cout << "Shared Vector addition time: " << ms2 << " ms" << endl;

	cout << "speed up: " << ms1 / ms2 << endl;





	delete[]A_h;
	delete[]B_h;
	delete[]C_h;
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	return 0;

}
