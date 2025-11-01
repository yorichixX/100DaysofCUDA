#include<iostream>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;

__global__ void vecMul(float* A, float* B, float* C, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n)
	{
		C[index] = A[index] * B[index];
	}

}

int main()
{
	int n = 600;
	size_t size = n * sizeof(float);

	float *A_h, *B_h, *C_h;

	 A_h = new float[n];
	 B_h = new float[n];
	 C_h = new float[n];

	for (int i = 0;i <= n-1;i++)
	{
		A_h[i] = i * 2;
		B_h[i] = i * 4;
	}

	float* A_d, * B_d, * C_d;
	cudaMalloc(&A_d, size);
	cudaMalloc(&B_d, size);
	cudaMalloc(&C_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
	int block = (n - 256 + 1) / 256;
	vecMul << <block, 256 >> > (A_d, B_d, C_d, n);
	cudaDeviceSynchronize();
	cudaMemcpy(C_h, C_d,size, cudaMemcpyDeviceToHost);
	

	for (int i = 0; i < 50; ++i)  //outputting just 50 elements from the resultant vector.
	{
		if (i % 10 == 0)
		{
			cout << endl;
		}
		cout << i+1 << ". " << C_h[i] << endl;
	}

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	delete[] A_h;
	delete[] B_h;
	delete[] C_h;


	return 0;
}
