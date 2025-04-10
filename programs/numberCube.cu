
#include <stdio.h>
#include <cuda_runtime.h>


// this is the kernel and it looks like a serial program because it is upto the GPU to make this parallel.
// __ is the way we define the kernel and it is void because it doesnot return anythings, instead it 
// writes the output in the pointers specified in the argument list
// cuda has the built in index called threadIdx which knows which thread it is running on i.e. index within a block


 
__global__ void cube(float *d_out, float *d_in){
    int idx = threadIdx.x; // since we launch 64 threads, the firss instance of threadIdx.x will return 0
    float f = d_in[idx];   // for each thread we are going to read the array element correspoing to this thread index from global memory and store it in the float variable f
    d_out[idx] = f * f * f; // after cubing we are going to write it back to global memory
}

int main(int argc, char **argv){
    const int ARRAY_SIZE = 96;
    const int ARRAY_BYTES =  ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    for (int i = 0; i< ARRAY_SIZE; i++){
        h_in[i] = float[i];
    }
    float h_out[ARRAY_SIZE];

    // declare GPU memory pointers just like we do for the CPU having the float * 
    float * d_in;
    float * d_out;

    // allocate the GPU memory
    // to tell the cuda that our data is actually on the GPU not the CPU 
    // we are using cudaMalloc with two arguments i.e. the pointer and the number of bytes to allocate
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    // it is like a regular Memcpy but it takes four arguments where it is destination, source and bytes from 
    // left and fourth arguement is the direction of the transfer
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    // cuda launc operator is indicated by three less than and greater than sign with some parameters in between
    // here it says launch the kernel name square on one block of 64 elements which takes two arguments
    // which are d_out and d_in. This code tells the CPU to launch on the GPU 64 copies of the kernel
    // on 64 threads 
    cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // copy back the result array to the CPU
    // Once we are done with the  kernel the results are in d_out on GPU and this cudaMemcpy will 
    // call will move memory from device to host and place it on h_out
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array

    for (int i= 0; i < ARRAY_SIZE; i++){
        printf("%f", h_out[i]);
        printf(((i%4) != 3) ? "\t" : "\n");
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;

}