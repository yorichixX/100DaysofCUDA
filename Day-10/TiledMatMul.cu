#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cmath>
using namespace std;

__global__ void tiledMatMul(float *A, float *B, float *C, int N, int K, int M)
{
    const int TILE_SIZE= 16;
    int row= blockIdx.y*TILE_SIZE + threadIdx.y;
    int column= blockIdx.x*TILE_SIZE + threadIdx.x; 

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for(int tile=0; tile<(K+TILE_SIZE-1)/TILE_SIZE; tile++)
    {
        int A_row= blockIdx.y*TILE_SIZE + threadIdx.y;
        int A_column= tile*TILE_SIZE + threadIdx.x;

        int B_row= tile*TILE_SIZE + threadIdx.y;
        int B_column= blockIdx.x*TILE_SIZE + threadIdx.x;
        //each thread loads 1 element from A and B each.
        if(A_row<N && A_column<K)
        {
            A_tile[threadIdx.y][threadIdx.x]= A[(A_row)*K + (A_column)];
        }
        else
        {
            A_tile[threadIdx.y][threadIdx.x]= 0.0f;
        }
        
        if(B_row<K && B_column<M)
            B_tile[threadIdx.y][threadIdx.x]= B[(B_row)*M +(B_column)];
        else
        {
            B_tile[threadIdx.y][threadIdx.x]= 0.0f;
        }
        __syncthreads();

        //compute phase
        for(int k=0;k<TILE_SIZE;k++)
        {
            sum += A_tile[threadIdx.y][k]*B_tile[k][threadIdx.x];
        }

        __syncthreads();  
        
    }

    if(row<N && column<M)
        {
            C[row*M + column]= sum;
        }
}

    int main()
    {
        int N= 512, M=512, K=512;
        const int TILE_SIZE=16;
        
        size_t sizeA= N*K*sizeof(float);
        size_t sizeB= K*M*sizeof(float);
        size_t sizeC= N*M*sizeof(float);

        float *A_h=(float*)malloc(sizeA);
        float *B_h=(float*)malloc(sizeB);
        float *C_h=(float*)malloc(sizeC);
        float *C_cpu=(float*)malloc(sizeC);

        float *A, *B, *C;
        cudaMalloc(&A, sizeA);
        cudaMalloc(&B, sizeB);
        cudaMalloc(&C, sizeC);

        for(int z=0;z<N*K;z++)
        {
            A_h[z]= static_cast<float>(z%7);
        }

        for(int x=0;x<K*M;x++)
        {
            B_h[x]= static_cast<float>(x%5);
        }

        cudaMemcpy(A,A_h,sizeA,cudaMemcpyHostToDevice);
        cudaMemcpy(B,B_h,sizeB,cudaMemcpyHostToDevice);
        
        dim3 threadsperBlock(TILE_SIZE,TILE_SIZE);
        dim3 blocksperGrid((M+TILE_SIZE-1)/TILE_SIZE,(N+TILE_SIZE-1)/TILE_SIZE);

        for(int i=0;i<N;i++)
        {
            for(int j=0;j<M;j++)
            {   
                float sum= 0.0f;
                for(int k=0;k<K;k++)
                {
                    sum += A_h[i*K + k]*B_h[k*M + j];
                }
            C_cpu[i*M+j]= sum;
            }
        }

        tiledMatMul<<<blocksperGrid,threadsperBlock>>>(A,B,C,N,K,M);

        cudaError_t err= cudaGetLastError();
        err=cudaDeviceSynchronize();

        if(err!=cudaSuccess)
            cout<<cudaGetErrorString(err)<<endl;
        
        cudaMemcpy(C_h,C,sizeC,cudaMemcpyDeviceToHost);

        bool correct = true;

        for(int i=0;i<N;i++)
        {
           for(int j=0;j<M;j++)
           {
                float diff= fabs(C_cpu[i*M + j]-C_h[i*M + j]);
                const float epsilon= 1e-3f;

                if(diff>epsilon)
                {
                    correct= false;
                    cout << "Mismatch at [" << i << "][" << j << "]\n";
                    cout << "CPU: " << C_cpu[i*M +j] << "GPU: " << C_h[i*M + j] << endl;
                    break;
                }
           } 
           if(!correct) break;
        }
        if(correct) cout<< "GPU result is correct!" <<endl;

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        free(A_h);
        free(B_h);
        free(C_h);
        free(C_cpu);

        return 0;
    }
    

