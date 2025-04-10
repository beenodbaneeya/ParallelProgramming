#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

//constants for matrix dimensions and tile sizes
const static int width = 4096; //width of the matrix
const static int height = 4096; // height of the matrix
const static int tile_dim = 16; // Tile dimension for shared memory


//Optimized transpose kernel with shared memory and padding to avoid bank conflicts
__global__ void transposed_SM_nobc_kernel(float *in, float *out, int width,  int height){
    //Declare a shared memory tile with padding
    __shared__ float tile[tile_dim][tile_dim + 1];  //Extra column for padding to avoid conflicts

    //Calculate the indices for the input matrix
    int x_tile_index = blockIdx.x * tile_dim;  //Start column of the tile in the input matrix
    int y_tile_index = blockIdx.y * tile_dim; //Start row of the tile in the input matrix

    //Compute global input index for the current thread
    int in_index = (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);

    //Load data from global memory into shared memory
    tile[threadIdx.y][threadIdx.x] = in[in_index];

    //synchronize threads to ensure all data is loaded into shared memory
    __syncthreads();

    // Calculate the indices for the output matrix ( tanspose the tile)
    int x_transpose_tile_index = blockIdx.y * tile_dim; //Transpose start column
    int y_transpose_tile_index = blockIdx.x * tile_dim; //Transpose start row

    //Compute global output index for the current thread
    int out_index = (x_transpose_tile_index + threadIdx.y) * height + (y_transpose_tile_index + threadIdx.x);

    //Write data from shared memory to global memory (transposed)
    out[out_index] = tile[threadIdx.x][threadIdx.y];

}

int main (){
    //Host matrices
    std::vector<float> matrix_in;
    std::vector<float> matrix_out;

    //Resize matrices to match the dimensions
    matrix_in.resize(width * height);
    matrix_out.resize(width * height);

    //Initialize input matrix with random values
    for (int i = 0; i < width * height; i++){
        matrix_in[i] = (float)rand() /(float)RAND_MAX;
    }

    //Device pointers
    float *d_in, *d_out;

    //Allocate memory on the device for input and output matrices
    cudaMalloc((void **)&d_in, width * height * sizeof(float));
    cudaMalloc((void **)&d_out, width * height * sizeof(float));

    //copy input matrix from host to device
    cudaMemcpy(d_in, matrix_in.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    //Kernel configuration
    int block_x = width / tile_dim;
    int block_y = height / tile_dim;

    dim3 grid(block_x, block_y); // Grid of blocks
    dim3 block(tile_dim, tile_dim); //Threads per block (tile_dim * tile_dim)


    //create CUDA events for timing
    cudaEvent_t start_kernel_event, end_kernel_event;
    cudaEventCreate(&start_kernel_event);
    cudaEventCreate(&end_kernel_event);

    //Warm up the GPU
    printf("Warming up GPU with Kernel execution...\n");
    for (int i = 0; i < 10; i++){
        transposed_SM_nobc_kernel<<<grid, block>>>(d_in, d_out, width, height);
    }

    //record start time
    cudaEventRecord(start_kernel_event,0);

    //Launch the transpose kernel multiple times for averaging 
    for (int i = 0; i < 10 ; i++){
        transposed_SM_nobc_kernel<<<grid, block>>>(d_in, d_out, width, height);
    }

    //Record end time and synchronize 
    cudaEventRecord(end_kernel_event,0);
    cudaEventSynchronize(end_kernel_event);

    //Calculate elapsed time
    float time_kernel;
    cudaEventElapsedTime(&time_kernel, start_kernel_event, end_kernel_event);


    //Print timing results 
    printf("Kernel execution complete.\n");
    printf("Execution time (average over 10 runs): %.6f ms\n", time_kernel / 10);
    printf("Bandwidth: %.6f GB/s\n", 2.0 * 10000 * (((double)(width) * (double)height) * sizeof(float)) /(time_kernel * 1024 * 1024 * 1024));

    //copy transposed matrix from device to host
    cudaMemcpy(matrix_out.data(), d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    //Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}