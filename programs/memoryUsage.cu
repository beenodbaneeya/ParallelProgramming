/* using local memory *
***********/

// a __device__ or __global__ function runs on the GPU
__global__ void use_local_memory_GPU(float in)
{
    float f; // variable "f" is in local memory and private to each thread
    f = in; // parameter "in" is in local memory and private to each thread
    // ... real code would presumably do other stuff here ...
    // (e.g., computations using 'f')
}

int main(int argc, char **argv)
{
    /*
    * First, call a kernel that shows using local memory
    */
    use_local_memory_GPU<<<1, 128>>>(2.0f);
    
    cudaDeviceSynchronize(); // Wait for GPU to finish
    return 0;
}

/*Explanation of Local Memory Usage:
Local Memory Definition:

Local memory is private memory for each thread

Variables declared inside a kernel function without any special memory qualifiers are by default placed in local memory

In the Example:

float f; declares a variable in local memory - each thread gets its own private copy

The parameter float in is also stored in local memory - each thread gets its own copy of the input value

Kernel Launch:

<<<1, 128>>> means we launch 1 block with 128 threads

Each of these 128 threads will have its own separate copies of f and in

Key Characteristics of Local Memory:

Not shared between threads

Has thread lifetime (created when thread starts, destroyed when thread ends)

Relatively slow (actually stored in global memory but cached in L1/L2)

Used for automatic variables that don't fit in registers

Common Uses:

Large local arrays or variables that can't fit in registers

Variables where the compiler determines register usage would be inefficient

Variables needed when register spilling occurs
*/


/* using global memory *
***********/
// a __global__ function runs on the GPU & can be called from host
__global__ void use_global_memory_GPU(float *array)
{
    // "array" is a pointer into global memory on the device
    array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}

int main(int argc, char **argv)
{
    /*
    * First, call a kernel that shows using local memory
    */
    use_local_memory_GPU<<<1, 128>>>(2.0f);
    
    /*
    * Next, call a kernel that shows using global memory
    */
    float h_arr[128];    // convention: h_ variables live on host
    float *d_arr;        // convention: d_ variables live on device (GPU global mem)

    // allocate global memory on the device, place result in "d_arr"
    cudaMalloc((void **) &d_arr, sizeof(float) * 128);
    
    // now copy data from host memory "h_arr" to device memory "d_arr"
    cudaMemcpy((void *)d_arr, (void *)h_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);
    
    // launch the kernel (1 block of 128 threads)
    use_global_memory_GPU<<<1, 128>>>(d_arr); // modifies the contents of array at d_arr
    
    // copy the modified array back to the host, overwriting contents of h_arr
    cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);
    
    // free device memory
    cudaFree(d_arr);
    
    // ... do other stuff ...
    return 0;
}


/*
Explanation of Global Memory Usage:
Global Memory Definition:

GPU's main memory space visible to all threads

Persists for the lifetime of the application

Allocated on the device but accessible from host code via CUDA API

In the Example:

float *array parameter in the kernel points to global memory

Each thread accesses its own index: array[threadIdx.x] = ...

threadIdx.x gives each thread a unique index (0-127 in this case)

Memory Management:

cudaMalloc: Allocates memory on the device (global memory)

cudaMemcpy: Copies data between host and device (note direction flags)

cudaFree: Releases device memory when done

Key Characteristics of Global Memory:

Shared among all threads (unlike local memory)

High latency but high bandwidth

Must be explicitly allocated and freed

Accessed via pointers from device code

Host-Device Interaction:

h_arr lives in host (CPU) memory

d_arr lives in device (GPU) global memory

Data must be explicitly copied between host and device

Common Uses:

Input/output data for kernels

Large datasets processed by GPU

Data shared between threads in different blocks

Persistent data used across multiple kernel calls
*/


/* using shared memory *
***********/
__global__ void use_shared_memory_GPU(float *array)
{
    // local variables, private to each thread
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[128];

    // copy data from global memory to shared memory
    sh_arr[index] = array[index];

    __syncthreads();    // ensure all writes to shared memory have completed

    // compute average of all previous elements
    for (i = 0; i < index; i++) { 
        sum += sh_arr[i]; 
    }
    average = sum / (index + 1.0f);

    // modify global memory based on the calculation
    if (array[index] > average) { 
        array[index] = average; 
    }

    // Note: This modification has no persistent effect:
    // sh_arr[index] = 3.14;  // Would vanish after kernel completes
}

//Explanation of Shared Memory Usage:
//Shared Memory Definition:

//Memory shared among all threads in a thread block

//Faster than global memory (on-chip memory)

//Declared with __shared__ qualifier

//Lifetime matches the thread block's execution

//Key Components in the Example:

//__shared__ float sh_arr[128]: Shared array visible to all 128 threads in block

//sh_arr[index] = array[index]: Each thread copies one element from global to shared memory

//__syncthreads(): Critical barrier that ensures all threads complete the copy

//Memory Hierarchy:

//array: Global memory (slow, persistent)

//sh_arr: Shared memory (fast, temporary)

//sum/average: Local memory (thread-private)

//Performance Characteristics:

//Shared memory is ~100x faster than global memory

//Enables thread cooperation within a block

//Limited size (typically 48KB per block)

//Common Patterns Shown:

//Staged computation: Load data from global → process in shared → write back

//Thread cooperation: All threads contribute to population of shared array

//Parallel reduction: Summation pattern (though simplified here)

//Important Notes:

//The __syncthreads() is crucial for correct shared memory usage

//Modifications to shared memory don't persist after kernel completes

//Global memory changes (array[index]) are permanent

//The commented-out sh_arr[index] = 3.14 has no effect (demonstrates shared memory's temporary nature)

//Typical Use Cases:

//Thread block-wide computations

//Data reuse patterns (when multiple threads access same data)

//Fast temporary storage for intermediate results

//Implementing parallel algorithms like reductions, scans, etc.


// calling the kernel using the shared memory is no different than calling the kernel that uses the global 
// memory as we can pass in a local variable that points global memory if we have allocated it.
// and what that kernel does with the host memory is not visibile to host code at all.
