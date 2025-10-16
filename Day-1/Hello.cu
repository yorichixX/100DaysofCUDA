#include <iostream>
__global__ void kernel(void) 
{
	printf("hello from GPU!\n");

}

int main(void)
{
	kernel<< <1, 1 >> > ();
	cudaDeviceSynchronize();
	std::cout << "Hello from CPU\n";
	return 0;
}
