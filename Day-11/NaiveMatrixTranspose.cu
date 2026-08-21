__global__ void float naiveMatTranspose(float *input , float *output, int rows, int cols)
{
    int row=  blockDim.y*blockIdx.y + threadIdx.y; 
    int col= blockDim.x*blockIdx.x + threadIdx.x;

    if(row<rows && col<cols)
    {
        output[col*rows + row] = input[row*cols + col];
    }
}
