#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
    printf("Hello world! I am thread in block %d\n", blockIdx.x);
}

int main(int argc, char **argv){
    //launch the kernel
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    //force the printf()s to flush
    cudaDeviceSynchronize();

    printf("That´s all!\n");

    return 0;
}


// Each different run of this program can produce 21 trillion outputs which is 16!