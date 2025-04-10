#include <iostream>
#include <cstdlib>

#define N 512  //Matrix size: N * N

__global__ void matrixMultiply(const float *A, const float *B, float *C , int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x

    if (row < width && col < width){
        float sum = 0.0;
        for (int k = 0; k < width; k++){
            sum+ = A[row * width + k] + B[ k * width + col];
        }
        C[row * width + col] = sum;
    }

}

//Host function to set up data and call the kernel
int main(){
    int size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    //Initialize matrices A and B with random values

    for (int i= 0; i < N * N; i++){
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float> (rand()) / RAND_MAX;
    }

    //Allocate memory on the device (GPU)

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy matrices A and B to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //Define grid and block dimensions
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((N + threadsPerBlock.x -1) / threadsPerBlock.x,(N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    //Launching the Kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A,d_B,d_C, N);

    //copy result back to host
    cudaMemcpy(h_c, d_C, size,cudaMemcpyDeviceToHost);

    //verify a few variables (optional)
    std:cout << "Matrix multiplication completed on GPU." << std::endl;

    //free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //free host memory
    free(h_A);
    free(h_B);
    free(h_c);

    return 0;

}