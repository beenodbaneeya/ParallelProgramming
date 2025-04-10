
# include <stdio.h>
#include <cuda_runtime.h>


// this is the kernel and it looks like a serial program because it is upto the GPU to make this parallel.
// __ is the way we define the kernel and it is void because it doesnot return anythings, instead it 
// writes the output in the pointers specified in the argument list
// cuda has the built in index called threadIdx which knows which thread it is running on i.e. index within a block


 
__global__ void square(float *d_out, float *d_in){
    int idx = threadIdx.x; // since we launch 64 threads, the firss instance of threadIdx.x will return 0
    float f = d_in[idx];   // for each thread we are going to read the array element correspoing to this thread index from global memory and store it in the float variable f
    d_out[idx] = f * f; // after squaring we are going to write it back to global memory
}

int main(int argc, char **argv){
    const int ARRAY_SIZE = 64;
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
    // on 64 threads   -- more explanation at the bottom of this page.
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

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


// commnad to compile this code - nvcc -o square square.cu
// instead of running the regular c compiler we are using nvcc which is nvidia c compiler where the output is going
// to be in the executable call square and the actual program name is square.cu


// square<<<1, ARRAY_SIZE>>>(d_out, d_in); - where 1 is the number of thread and 64 is the thread per block
// our hardware is capable of running many blocks at the same time and the maximum number of threads
// per block could be 512 in older GPUs and 1024 in newer GPUs
// example in our case if we want to square 128 instead of 64 we can change it to <<<1, 128>>>
// but if we want to run 1280 threads we can do it as shown below
// <<<10, 128>>>  or <<<5, 256>>> but we cannot do <<<1, 1280>>> because it will be lot for one block
// cuda supports 1,2 or 3D thread blocks and we can arrange them in 1,2 or 3D grid blocks



// we can actaully call our square with many blocks and grids as shown below:
// square<<<dim3(bx,by,bz), dim3(tx,ty,tz), shmem)>>>(...) where grid of blocks is bx.by.bz and block of threads is tx.ty.tz
// and shmem is the shared memory per block in bytes

// threadIdx - thread within block  (threadIdx.x  or threadTdx.y)
// blockDim - size of the block
// blockIdx - block within grid
// gridDim - size of grid



