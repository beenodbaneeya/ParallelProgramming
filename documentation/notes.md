# Important notes
Questions and answers:
1. What are 3 traditional ways HW designers make computers run faster?
faster clocks, more work/ clock cycle, more processors
2. Why are processors today getting faster?
because we have more transistors available for computation.
3. Which techniques are computer designers using today to build more power- efficient chips?
more, simpler chips.
4. We need to travel 4500 km . we have two options (cars that take two people at speed of 200 km/ hr) and (bus that take 40 people at 50 km /hr). calculate the latency and throughput for both of them.
car- latecny will be 22. 5 km when 4500 is divided by 200 but the throughput will be 2 people divided by 22.5  which will be 0.089 
bus - travelling 4500 km latency will be 90 hrs but the throughput will be 0.45 when 40 is 90.
Sometime, improved latency will increase the throuhput and vice verse. For GPU programming we are looking into throughput.

5. The basic GPU can do following things:
respond to CPU request to send data GPU->CPU
respond to CPU request to receive Data CPU->GPU
compute a kernel launched by CPU

6. What is the GPU good at?
- launching a large number of threads efficiently
- running a large number of threads in parallel

7. How many multiplications and times does this code takes if for 1 multiplication it is 2ns.
Here we have only one thread of execution i.e. one independent path of execution through the code. You can also see that the code has no explicit parallelism.

````c
for (i=0; i< 64; i++){
    out[i] = in[i] * in[i];
}
````

Multilpication - 64
execution time - 2 * 64 = 128ns

8. How does it work if launched in 64 instances of the same program if each multiplication takes 10ns ?
cpu launches 64 threads

multiplications - 64
execution time - 10 ns
for one single multiplication it takes 10 ns which seems to be higher but the overall throughput is very lowe compared to CPU programming.

9. answer the following for this kernel : kernel<<<dim3(8,4,2), dim3(16,16)>>>(...)
how many blocks - 64 (since it is 8 in x direction 4 in y and 2 in z direction )
how many thread per block - 256 (since it is 16 in x and 16 in y)
how many total threads - 16,384 (64 blocks of 256 threads)

10. check the problems that can be solved using the map i.e map(ELEMENTS, FUNCTION)
 the answer is - Add one to each element in an input array because it input is process in parallel.
 but we cannot sort an input array , because output is dependent on all the input array.Also we cannot sum all the all the elements in an input array neither computer the average of an input array.

11. Given a list of basketball players (name, height and rank in height(1st, 2nd tallest and so on))
write each palyers´s record into its location in a sorted list.
This is scatter operation as each thread is computing where to write its results.

12. how many times will a given input value  be read when applying each stencil?
2D von Neumann - 5
2D Moore - 9
3D von Neumann - 7

13. Label code snippets by communication pattern(Map, Gather, Scatter, Stencil, Transpose)

````c
float out[], in [];
int i = threadIdx.x;
int j = threadIdx.y;

const float pi = 3.1415;

out[i] = pi * in[i];      
// since there is one to one correspondance between output and the input, so it is Map operation

out[i + j*128] = in[j + i*128]; 
// It is also an one to one operation since one value in output is written in the output array
// corresponding to the every value that gets read from the input array. Array is represented in major 
// order in output and j major order in input. So this is the Transpose operation.

if (i % 2){
    //since only odd numbers thread are going to execute this, it is not map, and also it is not
    // transpose.We cant say it stencil too because, it should generate a result for every element
    // in the output array and it does not do that.Here the thread is taking the input at a given location
    // multiplying it by pi and plcaing into a couple of different places in the output array .i.e. 
    // increamenting a couple of different places in output array. so it is a Scatter operation.
    out[i-1] += pi * in[i]; out[i+1] += pi * in[i];

    // This is the gather because every threads is writing a single location in the output array.
    // And it is reading from multiple places in the input array.Note that it looks line a stencil 
    // operation since it is reading from local neighborhood, doing some averaging and writing the result
    // we cannot say it stencil because it is not writing into every location because of the if statement.

    out[i] = (in[i] + in[i-1] + in[i+1]) * pi/ 3.0f;
}
````

14. Parallel Communication Pattern
Map - one to one
Transpose - one to one
Gather - many to one
Scatter - one to many
stencil - several to one
reduce - all to one
scan/sort - all to all


15. Which of them are true and false
A thread blocks contains many threads - true
An SM may run more than one block - true
A block may run on more than one SM - False
All the threads in a thread block may cooperate to solve a problem - true
All the threads that run on a given SM may cooperate to solve a problem - false

16. Which of them are true or false
Programmer or GPU is responsible for defining thread blocks in software
answer - programmer
programmer or GPU is responsible for allocating thread blocks to hardware streaming multiprocessors(SMs)
answer - GPU

17. Given a single kernel that is launched on many threads blocks including X,y,Z, the programmer can specify
a. that block X will run at the same time as block Y
b. that the block X will run after block Y
c. that block X will run on SM Z

answer - all of them are false.

18. Which of them are true or false
a. all threads from a block can access the same varaible in that blocks shared memory
b. threads from two different blocks can access the same variable in the global memory
c. threads from different blocks have their own copy of local varaibles in local memory
d. threads from the same block have their own copy of local varaibles in local memory

All of them are true.

19. Barrier - point in the program where threads stops and wait  and when all the threads reached the barrier , they can proceed.
The need for the barries in the code given below:
````c
int idx = threadIdx.x;

__shared__ int array[128];
array[idx] = threadIdx.x;
if(idx < 127)
   array[idx] = array[idx+1]
````

This will need 3 barries as shown below
````c
int idx = threadIdx.x;

__shared__ int array[128];

// this operation is write so we need barrier after this write completes to make sure nobody tries ready
//  from the array until all of the threads have finished the write operation
array[idx] = threadIdx.x;
__syncthreads();
if(idx < 127)
   // array[idx] = array[idx+1] - commented because it needs restructing, all of the reads on the right
   // should be done before writing in the array.
   int temp = array[idx+1];
   __syncthreads();
   array[idx] = temp;
   __synchthreads(); // before anybody reads from it

````

20. Are the following code snippets correct?
````c
__global__ void foo(){
    __shared__ int s[1024];
    int i = threadIdx.x;

    __syncthreads();
    s[i] = s[i-1]; //this is wrong as there is no guarantee that the reads on the right complete before
    // write on te left.the correct way to write it would be to write it in the temp varaiable use
    // syncthread() and then write from temp to s[i]

    __syncthreads();
    if (i%2) s[i] = s[i-1]; //here only the odd threds tries to writes so there is no problem. and this
    // is corrrect

    __syncthreads();
    s[i] = (s[i-1] + s[i] + s[i+1]) / 3.0; // this is just like first one, in order to make sure all of
    // these reads on right happen before write on left , we need to use temp like before,after reading
    // and also writing before printing it.
    printf("s[%d] == %f \n", i, s[i]);
    __syncthreads();
}
````

21. Writing efficient programs
a. maximize the arithmetic intensity i.e. math/memory which means 
- maximize the compute ops per thread
- minimize the time spent on memory per thread (we can do that by moving frequently accessed data to fast memory local>share>global>cpu host)


22. Number the operations from fasted to slowest
````c
__global__ void foo(float *x , float *y, float *z){
    __shared__ float a,b,c;
    float s,t,u;

    s = *x;     - 3 since it is reading from global memory and putting into local vaariable
    t = s;      - 1  since it is local
    a = b;      -2 since it is in the shared memory
    *y = *z;    -4  since it is in global memory
}
````

23.  Which of the statements have coalesced access pattern?

````c
__global__ void foo(float *g){
    float a = 3.14;
    int i = threadIdx.x;
}

g[i] = a; // every thread is simply accessing allocation and memory defined by its thread index.
         // since bunch of threads will access its adjacent contiguous location in memory, Hence,
         // coalesced
g[i*2] = a; // every thread is accessing a location in memory defined by its index times 2.Hence 
        // Strided access

a = g[i] // just like first, instead of read we are doing write. hence coalesced

a = g[BLOCK_WIDTH/2 + i];  // every thread is reading from the location defined by g plus some offset 
                            // block width over 2 plus the thread index. for any of the block width, 
                            // every thread will be accessing adjacent locations starting at that 
                            // offset, Henced, coalesced.

g[i] = a * g[BLOCk_WIDTH/2 + i]; // we are reading from this location which is defined by an offset plus
                            // the thread index.we will multiply it with constant and store results back 
                            // to contiguos chunck of memory.coalesced read followed by coalesced write.

g[BLOCK_WIDTH-1 - i ] = a; // even though we are doing subsstraction , it is accessing a contiguous 
                            // Hence coalesced.
````

