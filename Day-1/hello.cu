#include <iostream>
__global__ void kernel(void) 
{
	printf("hello from GPU!\n"); //this part runs on GPU

}

int main(void)
{
	kernel<< <1, 1 >> > (); 
	cudaDeviceSynchronize();
	std::cout << "Hello from CPU\n"; //this part runs on CPU
	return 0;
}
