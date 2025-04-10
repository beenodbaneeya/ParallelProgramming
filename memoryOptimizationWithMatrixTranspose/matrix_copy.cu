#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

//constants for matrix dimensions and tile sizes
const static int width = 4096;  //width of matrix
const static int height = 4096; //Height of matrix
const static int tile_dim = 16; //Tile size (block dimensions)

//kernel function to perform simple data copy from one array to another
__global__ void copy_kernel(float *in, float *out, int width, int height){
    //compute global thread indices
    int x_index = blockIdx.x * tile_dim + threadIdx.x; //column index
    int y_index = blockIdx.y * tile_dim + threadIdx.y; //row index

    // compute the linear index for 2D matrix
    int index = y_index * width + x_index;

    // copy the element from input array to output array 
    out[index] = in[index];
}


int main(){
    //Host-side vectors for input and output matrices
    std::vector<float> matrix_in;
    std::vector<float> matrix_out;

    //Resize host vectors to match the matrix dimensions
    matrix_in.resize(width * height);
    matrix_out.resize(width * height);

    //Initialize the input matrix with random float values
    for (int i=0; i < width * height; i++){
        matrix_in[i] = (float)rand() / (float)RAND_MAX;
    }

    //Device pointer for input and output matrices
    float *d_in, *d_out;

    //Allocate memory on the device for input and output matrices
    cudaMalloc((void **)&d_in, width * height * sizeof(float));
    cudaMalloc((void **)&d_out, width * height * sizeof(float));
    
    //copy the input matrix data from host to device
    cudaMemcpy(d_in, matrix_in.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);


    printf("Setup complete. launching kernel \n");

    //compute the grid  and block dimensions
    int block_x = width / tile_dim;  //Number of blocks in x-dimension
    int block_y = height / tile_dim; //Number of blocks in y-dimension

    //Create CUDA events for timing kernel execution
    cudaEvent_t start_kernel_event, end_kernel_event;
    cudaEventCreate(&start_kernel_event);
    cudaEventCreate(&end_kernel_event);

    //Warm up the GPU by launching the kernel  multiple times (optional, improves timing accuracy)
    printf("Warming up the GPU...\n");
    for (int i = 1; i <= 10; i++){
        copy_kernel<<<dim3(block_x, block_y), dim3(tile_dim, tile_dim)>>>(d_in, d_out, width, height);
    }

    //Start recording the kernel execution time
    cudaEventRecord(start_kernel_event, 0);
    
    //launch the kernel multiple times to calculate the average execution time
    for (int i = 1; i <= 10; i++) {
        copy_kernel<<<dim3(block_x, block_y), dim3(tile_dim, tile_dim)>>>(d_in,d_out, width, height);
    }

    //Stop recording the kernel execution time
    cudaEventRecord(end_kernel_event, 0);
    cudaEventSynchronize(end_kernel_event);

    // Synchronize the device to ensure all kernels have completed execution
    cudaDeviceSynchronize();

    //calculate the elapsed time for the kernel
    float time_kernel;
    cudaEventElapsedTime(&time_kernel, start_kernel_event, end_kernel_event);

    printf("Kernel execution complete. \n");
    printf("Event timings: \n");
    printf(" %.6f ms - copy \n", time_kernel / 10);

    //calculate memory bandwidth
    double bandwidth = 2.0 * ((double)width * (double)height * sizeof(float)) / (time_kernel * 1024 * 1024 * 1024);
    printf("Bandwidth: %.6f GB/s\n", bandwidth);

    //copy the output matrix from device to host
    cudaMemcpy(matrix_out.data(), d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    //Free device memory
    cudaFree(d_in);
    cudaFree(d_out);


    //Destroy CUDA events 
    cudaEventDestroy(start_kernel_event);
    cudaEventDestroy(end_kernel_event);

    printf("Execution complete. Check results for correctness. \n");

    return 0;



}
