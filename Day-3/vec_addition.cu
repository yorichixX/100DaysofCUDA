#include<iostream>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;


__global__ void vec_add_kernel(float* A, float* B, float* C, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		C[idx] = A[idx] + B[idx];
	}
}
 


void vec_add(float *A_h, float *B_h, float *C_h, int n)
{
	int size = n * sizeof(float);
	float *A_d, *B_d, *C_d;

	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);
	cudaMalloc((void**) &C_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
	int blocks = (n)/256;
	vec_add_kernel << < blocks, 256 >> >(A_d,B_d,C_d,n);
	cudaDeviceSynchronize();
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

}

int main()
{
	int n = 1 << 20;
	float* A = new float[n];
	float* B = new float[n];
	float* C = new float[n];

	for (int i = 0;i < n;++i)
	{
		A[i] = i;
		B[i] = 2 * i;
	}

	vec_add(A, B, C, n);

	for (int i = 0; i < 100; ++i) // showing the first 100 added elements from C
		cout << i+1 <<". " << C[i] << " " << endl;
	cout << endl;

	cout << "The last element of the vector is: " << C[(1 << 20) - 1] << endl;

		delete[] A;
		delete[] B;
		delete[] C;
}
