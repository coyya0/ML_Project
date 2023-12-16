#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 32

// GPU code for dot product of matrix (A) and matrix(B) -- none squared metrix
__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    // TODO 1
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum =0;
	if( col < k && row < k)
	{
		for(int i=0; i<n ; i++)
		{
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
} 

// GPU code for dot product of matrix (A) and matrix(B) -- squared metrix
__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n) 
{
    // TODO 2
	__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
	int idx;

	for (int sub = 0; sub< gridDim.x; ++sub)
	{
		idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
		if(idx >= n*n)
		{
			// n may not divisible by block_size
			tile_a[threadIdx.y][threadIdx.x] = 0;
		}
		else
		{
			tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
		}
		idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
		if(idx >=n*n)
		{
			tile_b[threadIdx.y][threadIdx.x] = 0;
		}
		else
		{
			tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
		}
        __syncthreads();

        for (int k =0; k< BLOCK_SIZE; k++)
        {
            tmp += tile_a[threadIdx.y][k]* tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}



__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols) 
{
    // TODO 3
    unsigned int idx = blockIdx.x * blockDim.x +threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y +threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}




// CPU code for dot product of matrix (A) and matrix(B) 
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) 
{
    // TODO 4
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for(int h = 0; h<n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
           }
           h_result[i * k + j] = tmp;
        }
    }
}


int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    printf("Dot Product of matrix A(m x n) and matrix B(n x k)\n");
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM, h_cc is used to store CPU result
        // TODO 5
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **)&h_a, sizeof(int)*m*n); // m*n matrix
    cudaMallocHost((void **)&h_b, sizeof(int)*n*k); // n*k matrix
    cudaMallocHost((void **)&h_c, sizeof(int)*m*k); // m*k matrix
    cudaMallocHost((void **)&h_cc, sizeof(int)*m*k); // m*k matrix

    // random initialize matrix A
        // TODO 6
    printf("matrix A : \n");
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){ //m*n ma
            h_a[i * n + j] = 3*i+j+1;
            printf("%d\t ", h_a[i*n +j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("matrix B : \n");
    h_b[0] = 1;
    h_b[1] = -2;
    h_b[2] = 3;
    h_b[3] = 3;
    h_b[4] = 5;
    h_b[5] = 2;
    h_b[6] = -1;
    h_b[7] = 3;
    h_b[8] = -4;

    
        
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){ //m*n ma                   
            printf("%d\t ", h_b[i*n +j]);
        }
        printf("\n");
    }
    printf("\n");
 
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms; 
    // some events to count the execution time
        // TODO 8
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start to count execution time of GPU version
        // TODO 9
    cudaEventRecord(start,0);
    // Allocate memory space on the device 
        // TODO 10
    int *d_a, *d_b, *d_c,*t_b;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);
    cudaMalloc((void **) &t_b, sizeof(int)*n*n);

    // copy matrix A and B from host to device memory
        // TODO 11
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);
    cudaMemcpy(t_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);


    // initialize GPU kernel
        // TODO 12
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
   
    // Launch GPU kernel 
    if(m == n && n == k)
    {
        // TODO 13

        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a,d_b,d_c,n); 
    }
    else
    {
        // TODO 14
        
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c,m,n,k);
    }

    // Transefr results from device(GPU) to host(CPU) 
        // TODO 15
    cudaMemcpy(h_c,d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // time counting terminate
        // TODO 16
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
        // TODO 17
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start,stop);
    // start the CPU version
         // TODO 18
    cudaEventRecord(start,0);

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);

    // validate results computed by GPU
        // TODO 19
    int all_ok = 1;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < k; ++j)
        {
            printf("div_cpu[%d][%d] : %d , div_gpu[%d][%d] : %d ", i, j, h_cc[i*k +j], i, j, h_c[i*k +j]);
            if(h_cc[i*k +j] != h_c[i*k +j]);
            {
                all_ok =0;
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);
   
    //compute speedup
    if(all_ok)
    {
        printf("results are correct!!!, GPU speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results... suggest to change the BLOCK_SIZE !! \n");
    }

    // free memory
        // TODO 20
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
