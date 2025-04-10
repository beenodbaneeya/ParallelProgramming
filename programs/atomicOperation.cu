#include <stdio.h>
#include <gputimer.h>

#define ARRAY_SIZE 10
#define NUM_THREADS 1000000
#define BLOCK_WIDTH 1000  // number of blocks will be 1000000/1000 = 1000 blocks

void print_array(int *array, int size)

__global__ void increment_atomic(int *g){
    //calculate which thread this is
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread increments consecutive elements, wrapping at ARRAY_SIZE
    i = i % ARRAY_SIZE;
    atomicAdd(&g[i], 1);  // Atomic increment operation
}

int main(int argc, char**argv){
    GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
            NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // Host Memory
    int h_array[ARRAY_SIZE];
    conts int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // Device memory
    int *d_array;
    cudaMalloc((void **)&d_array, ARRAY_BYTES);
    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    // Launch Kernel
    timer.start();
    increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.stop();

    // copy results back
    cudaMemcpy(h_array,d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);
    printf("Time elapsed = %g ms\n", timer.Elapsed())

    //cleanup
    cudaFree(d_array);
    return 0;

}



// We use atomic operation to get rid of this problem. for instance if we write the kernel like this
__global__ void increment_naive(int *g){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % ARRAY_SIZE;
    g[i] = g[i] + 1;
}
// This will create the random output because each thread want to write at the same time. so there is no guarantee that the updated value is read by the thread each time.
// Race Condition Problem: Without atomic operations, when multiple threads try to modify the same memory location simultaneously, you get a race condition. Some increments might be lost because threads overwrite each other's changes.
// Atomic Operations Solution: atomicAdd(&g[i], 1) guarantees that: The read-modify-write operation happens as one indivisible unit, No other thread can interfere during the operation, All increments will be counted correctly
