__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    const int TILE_SIZE= 16;

       __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (row < rows && col < cols)
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    int out_col = blockIdx.y * TILE_SIZE + threadIdx.x;
    int out_row = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (out_row < cols && out_col < rows)
        output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
}
