#include <stdio.h>
#include <cuda_runtime.h> // gives access to useful CUDA functions like cudaMalloc, cudaMemcpy and others

//kernel to convert RGBA to GreyScale
//const uchar4* d_rgbaImage → GPU pointer to the input color image (RGBA format, one pixel = 4 bytes).unsigned char* d_greyImage → GPU pointer to output grayscale image (one byte per pixel).
// int numRows, int numCols → dimensions of the image.uchar4 is a CUDA built-in type like a struct with .x, .y, .z, .w representing RGBA (Red, Green, Blue, Alpha


__global__ void rgba_to_greyscale(const uchar4* d_rgbaImage, unsigend char* d_greyImage, int numRows, int numCols){
    //This calculates the pixel coordinates that each thread will process.This maps the 2D CUDA grid/block/thread layout to 2D image coordinates (x = column, y = row).
    int x = blockIdx.x * blockDim.x + threadIdx.x;  //column
    int y = blockIdx.y * blockDim.y +  threadIdx.y; // row

    // Prevents threads from accessing invalid memory (pixels outside the image).essential when your grid size may cover more threads than pixels (like padding).

    if ( x >= numCols || y >= numRows)
        return;

    // in a 2D image stored in 1D array, row-major order is used:index = row * width + column
    int idx = y * numCols + x;  //Converts the 2D (x, y) location to a 1D index so we can access memory using a linear array.

    uchar4 rgba = d_rgbaImage[idx]; // Fetches the uchar4 pixel at this thread’s position.

    float grey = 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;

    // Casts the float result to a unsigned char (0–255 intensity) and stores it back into the output grayscale image.
    // static_cast<unsigned char>(grey) safely converts the float to byte
    d_greyImage[idx] = static_cast<unsigend char>(grey);

}


int main(){
    const int numRows = 4;
    const int numCols = 4;
    const int numPixels = numRows * numCols; // total pixel is 16

    // sizeof(uchar4) gives the size of the uchar4 data type, which is 4 bytes (1 byte for each of the RGBA components).
    size_t rgbaSize = numPixels * sizeof(uchar4);  // This calculates the size required to store the RGBA image in memory.
    // unsigned char is a data type representing a single byte, which is used to store the grayscale value (0 to 255).
    size_t greySize = numPixels * sizeof(unsigend char); // This calculates the size required to store the grayscale image in memory.

    //host image arrays
    uchar4 h_rgbaImage[numPixels];   // This declares an array named h_rgbaImage on the host (CPU) side, which will store the RGBA image.
    unsigend char h_greyImage[numPixels]; // This declares an array named h_greyImage on the host (CPU) side, which will store the grayscale image.


    // Fill in sample values RGBA
    // red and green values increases while The Blue (B) value decreases as the index increases (255 - i * 10), giving a reverse gradient effect.
    for (int i=0; i<numPixels; i++){
        h_rgbaImage[i].x = i * 10; // R
        h_rgbaImage[i].y = i * 5;       // G
        h_rgbaImage[i].z = 255 - i * 10; // B
        h_rgbaImage[i].w = 255;         // A
    }

    //device  pointers
    uchar4*  d_rgbaImage;
    unsigend char* d_greyImage;

    // Allocate device memory
    cudaMalloc(&d_rgbaImage, rgbaSize);
    cudaMalloc(&d_greyImage, greySize);

    // copy host data to device
    cudaMemcpy(d_rgbaImage, h_rgbaImage, rgbaSize, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 blockSize(16, 16);    // each block contains a total of 16 * 16 = 256 threads (16 threads per row × 16 rows).
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
                  (numRows + blockSize.y - 1) / blockSize.y);
    


    // launch the kernel
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_greyImage, d_greyImage, greySize, cudaMemcpyDeviceToHost);

    // Print grayscale output
    printf("Grayscale image:\n");
    for (int i = 0; i < numPixels; ++i) {
        printf("%3d\t", h_greyImage[i]);
        if ((i + 1) % numCols == 0) printf("\n");
    }


    // Free device memory
    cudaFree(d_rgbaImage);
    cudaFree(d_greyImage);

    return 0;

}


























// how pixels are represented - most common method is to specify how much red, blue and green is presented in each pixel
// each color is called channel.if it is 0 it is fully black and if it is 255 , then it is fully white
// in cuda each pixel is represented by uchar4 struct  which has four unsigend char components namely x(red),y(green),z(blue) and w(aplha -which carry transparency information).

// converting color to black and white one way is to divide the sum of RGB channels by 3. i.e. I = (R+G+B)/3
// for the incentivity of the color we are  going to multiply each channel with different numbers as shown below
// I = .299f * R + .587f * G + .114f * B 

//Example Calculation:
//For an image of size 6×4 (6 columns and 4 rows), with:

//blockDim.x = 3 (3 threads per block in the x-direction)

//blockDim.y = 2 (2 threads per block in the y-direction)

//gridDim.x = 2 (2 blocks in the x-direction)

//gridDim.y = 2 (2 blocks in the y-direction)

//We have:2×2 = 4 blocks in total.Each block processes 3×2 = 6 threads.


// dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
//              (numRows + blockSize.y - 1) / blockSize.y);

// numCols is the total number of columns (width of the image).

//blockSize.x is the number of threads per block in the x-direction (16 threads per block in this case).

//(numCols + blockSize.x - 1): The reason for adding blockSize.x - 1 is to ensure that any extra portion of the image (if it doesn't fit evenly into the blocks) will still get a full block of threads. Essentially, this performs round-up division.

//For example, if numCols = 30, and blockSize.x = 16:

//(30 + 16 - 1) = 45, so 45 / 16 = 2 blocks will be needed.

// numRows is the total number of rows (height of the image).

//blockSize.y is the number of threads per block in the y-direction (16 threads per block in this case).

//(numRows + blockSize.y - 1): Similar to the x-direction, adding blockSize.y - 1 ensures that any leftover rows that don't fill a full block still get a full block of threads.

//For example, if numRows = 35, and blockSize.y = 16:

//(35 + 16 - 1) = 50, so 50 / 16 = 3 blocks will be needed.