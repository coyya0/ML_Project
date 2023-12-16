#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "common.h"
#include "common_string.h"
#define BLOCK_SIZE 32
#define MAX_KERNEL_WIDTH 10



// GPU code for dot product of matrix (A) and matrix(B) -- none squared metrix
__constant__ float M[MAX_KERNEL_WIDTH * MAX_KERNEL_WIDTH];
__global__ void gpu_matrix_mult(int *in,int *out,int *c, int m, int n, int k)
{ 
    // TODO 1
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum =0;
	if( col < k && row < m)
	{
		for(int i=0; i<n ; i++)
		{
			sum += in[row * n + i] * out[i * k + col];
		}
		c[row * k + col] = sum;
	}
} 

__global__ void gpu_matrix_convolution(float* in, float* out,  int width, int kernel_width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockDim.y*blockIdx.y + ty;
    int col = blockDim.x*blockIdx.x + tx;

 
    if (col < width-kernel_width+1 && row < width-kernel_width+1 ) {
        //여기 문제는 아닌거같다
        int row_start_point = row ;
        int col_start_point = col ;
        //start point x,y <= width - kernel_width 
        float val = 0.f;
        if(row_start_point <= (width-kernel_width )&& col_start_point<=(width - kernel_width) ){
            // filter가  input 밖으로 나가는 것을 제한 8*8 matrix 에서 3*3 kernel은 5*5 만큼 이동 
            for (int i = 0; i < kernel_width; i++) {// 3*3 kernel 만큼 시행
                for (int j = 0; j < kernel_width; j++) {
                    int row_idx = row_start_point + i;
                    int col_idx = col_start_point + j;
                    if (row_idx >= 0 && row_idx < width && col_idx >= 0 && col_idx < width) {
                        val += in[row_idx*width + col_idx]* M[i*kernel_width + j];
                        /*printf("in_index[%d] = %f M_index[%d] = %f  val=%f\n",row_idx*width+ col_idx,in[row_idx*width + col_idx],
                        i*kernel_width + j,  M[i*kernel_width + j],val);     */               
                    }
                }
            }
        }
  
        out[row*(width-kernel_width+1) + col] = val;
        //(width-kernel_width+1) 이부분에서 width로 해서 오류가 남 output_width 잘못계산!!
    }
    __syncthreads();
    
}

/*__global__ void gpu_matrix_convolution(float *input, float *kernel, float *output, int i_row, int k_row, int result_size)
{	
    // TODO 2
    //blockIdx.x ( 0 ~ 2) , blockIdx.y ( 0 ~ 32)
    //blockIdx : 그리드 내 블럭번호
    //blockDim : 블럭 안의 쓰레드 크기(수)
    //threadIdx : block 안의 쓰레드번호 , Tx,Ty 로 이루어짐 
	__shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];

 
    int idx_a=0;

    int col_o = threadIdx.x + BLOCK_SIZE*blockIdx.x;
    int row_o = threadIdx.y + BLOCK_SIZE*blockIdx.y; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;


 


	for (int sub = 0; sub< gridDim.x; ++sub)
	{
		//step 1: idx = row * n + sub * BLOCK_SIZE + threadIdx.x; n -> m
        idx_a = row_o * i_row + sub * BLOCK_SIZE + threadIdx.x;
        //target tile_a <= d_a
		if(idx_a >=i_row * i_row) //step 2: n*n -> m*m
		{
			// n may not divisible by block_size
			tile_a[threadIdx.y][threadIdx.x] = 0;
		}
		
        else
        {
			tile_a[threadIdx.y][threadIdx.x] = input[idx_a];         
		}
        //target tile_b <= d_b
		
        __syncthreads();
        // tile_a = input , tile_b = kernel and  sync

        float val = 0.f;
    if (ty < BLOCK_SIZE  && tx < BLOCK_SIZE  ) {
        for (int i = 0; i < k_row; i++) {
            for (int j = 0; j < k_row; j++) {
                val += tile_a[i+ty][j+tx] * kernel[i*k_row + j];
            }
        }
 
        if (row_o < result_size && col_o < result_size)
            output[ row_o*result_size + col_o] = val;
        //printf("Tx[%d] Ty[%d] k[%d] tile_a[%d][%d] = %f tile_b[%d][%d] = %f tmp = %f \n",threadIdx.x, threadIdx.y, k, 0, threadIdx.x,a, 0,threadIdx.x,b,tmp);
    }
       
        __syncthreads();
    }
    /*if(Ty < result_size && Tx < result_size)//step 4 row,col< n - > row,col< k  여기서 8까지 출력하던게 20으로 늘어남 
    {
        output[Ty * result_size + Tx] = sum; //step 5 row * n -> row * k 20까지 출력하던게 36으로 늘어남
        //왜 20일까..? row * 3 + col 이면 최대 24를 출력해야하는데 이유를 모르겠다        
    }
}
*/
void cpu_matrix_convolution(float *h_a, float *h_kernel, float *h_result, int m, int n, int k)
{
    //h_a : input , h_kernel : kernel ,h_result : feature_map
    //m = 8 , n = 3 , k = 6   
              

    
    // m: input , n : filter, k = m - n + 1 : feature_map

    printf("===========cpu convolution matrix============\n");
    for (int x = 0; x < k; x++) {// feature_map* feature_map
        for(int y = 0; y < k; y++){ 
            float sum=0.0;
            for (int row = 0; row < n; row++) { //filter * filter 
                float tmp=0.0;               
                for (int col = 0; col < n; col++) { //m*n ma
                    int idx_a = row*m+col + x*k+y +(n-1)*x; // x*k+y : 오른쪽으로 이동 , 2*x : 행이동 
                    int idx_b = row*n+col;
                    tmp = h_a[idx_a] * h_kernel[idx_b];                                   
                    sum += tmp;   
                }  
            }          
            h_result[x*k+y] = sum;  
            printf("%f\t ", h_result[x*k+y]);  
        }
        printf("\n");
    }
}

// CPU code for dot product of matrix (A) and matrix(B) 
void cpu_matrix_mult(float *h_a, float *h_kernel, float *h_result, int m, int n, int k) 
{
    // TODO 4
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float tmp = 0.0;
            for(int h = 0; h<n; ++h)
            {
                tmp += h_a[i * n + h] * h_kernel[h * k + j];
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
    printf("Dot Product of matrix input(m x n) and matrix filter(n x k)\n");
    printf("please type in input : m , filter : n\n");
    scanf("%d %d", &m, &n);
    // allocate memory in host RAM, h_cc is used to store CPU result
           // TODO 5

    
    float *h_a, *h_kernel, *h_c, *h_cc;
    h_kernel = (float*)malloc(m*m*sizeof(float));
    k = m - n + 1 ;
    cudaMallocHost((void **)&h_a, sizeof(float)*m*m); // m*m matrix    
    cudaMallocHost((void **)&h_c, sizeof(float)*k*k); // k*k matrix
    cudaMallocHost((void **)&h_cc, sizeof(float)*k*k); // k*k matrix 
    

    // random initialize m*m input matrix 
        // TODO 6
    
    printf("input matrix : \n");
   
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < m; ++j){   
            h_a[i * m + j] = rand() / (float)RAND_MAX;        
            printf("%f [%d]\t ", h_a[i*m +j],i * m + j);
        }
      
        printf("\n");
    }
    printf("\n");



    // random initialize n*n filter matrix 
        // TODO 7
    printf("filter matrix : \n");
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){   
            h_kernel[i * n + j] = rand() / (float)RAND_MAX;         
            printf("%f [%d]\t ", h_kernel[i*n +j], i*n +j);
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
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*m*m);
    cudaMalloc((void **) &d_b, sizeof(float)*n*n);
    cudaMalloc((void **) &d_c, sizeof(float)*k*k);
    CUDA_CHECK(cudaMemcpyToSymbol(M, h_kernel, n*n*sizeof(float)));
    // copy matrix A and B from host to device memory
        // TODO 11
    cudaMemcpy(d_a, h_a, sizeof(float)*m*m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_kernel, sizeof(float)*n*n, cudaMemcpyHostToDevice);


    // initialize GPU kernel
        // TODO 12
    unsigned int grid_rows = (m + BLOCK_SIZE -1 ) / BLOCK_SIZE; // k*k 만큼 연산을 해야함
    unsigned int grid_cols = (n + BLOCK_SIZE -1 ) / BLOCK_SIZE;
    unsigned int grid_conv = (k + BLOCK_SIZE) / BLOCK_SIZE;
    dim3 dimGrid_conv(grid_conv,grid_conv);
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    
    //dimGrid = (2 , 5) dimBlock = (32,32)
    // Launch GPU kernel 
    
    // TODO 13
    // (2,5) , (32,32) gridDim.x = 2, gridDim.y = 5
    //<<<block 수 (gridDim), thread 수 (blockDim)>>> = > <<<10 , 1024>>> 10 block , 1024 thread    
    
    
    gpu_matrix_convolution<<<dimGrid_conv,dimBlock>>>(d_a,  d_c, m, n );
    // dimGrid_conv * dimBlock 만큼 시행
 
   

    // Transefr results from device(GPU) to host(CPU) 
        // TODO 15
    cudaMemcpy(h_c,d_c, sizeof(float)*k*k, cudaMemcpyDeviceToHost);
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
    printf("cpu_matrix_convolution\n"); // core dumped error , 수정완료
    
    cpu_matrix_convolution(h_a, h_kernel, h_cc, m, n, k);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    printf("cudaEventElapsedTime\n");
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    
    printf("===========gpu convolution matrix============\n");
    for (int i = 0; i < m - n+1; i++){
        for (int j = 0; j <m - n+1; j++){
                printf("%f\t ", h_c[i*(m - n+1)+j]);  
        }
        printf("\n");
    }
    // validate results computed by GPU
        // TODO 19
    int all_ok = 1;
    //square_conv
    for(int i = 0; i < k; ++i)
    {
        for(int j = 0; j < k; ++j){
            printf("cpu[%d][%d] : %f == gpu[%d][%d] : %f  , i*k+j = [%d]", i, j, h_cc[i*k +j], i, j, h_c[i*k +j],i*k+j);
            if(h_cc[i*k +j] != h_c[i*k +j]);
                {
                   all_ok =0;
                }
               printf("\n");
            }
        printf("\n");
    }
   

    printf("Time elapsed on matrix convolution of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    printf("Time elapsed on matrix convolution of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);
   
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
    cudaFreeHost(h_kernel);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
