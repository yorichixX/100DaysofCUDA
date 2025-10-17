#include<iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;

__global__ void Hello()
{
	printf("Hello from GPU %d \n", threadIdx.x);

}

int main()
{
	cout << "Hello from CPU" << endl;
	Hello << <3, 2 >> > ();

	cudaDeviceSynchronize();
	return 0;

}
