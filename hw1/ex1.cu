#include "ex1.h"

#define THREADS_NUM TILE_WIDTH*6
#define COMMON_SIZE 256

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    extern __shared__ int temp[];
    for (int stride = 1; stride < arr_size; stride *= 2) {
        for (int i = tid; i < arr_size; i += num_threads) {
            if (i >= stride) {
                temp[i] = arr[i - stride];
            }
        }
        __syncthreads();
        for (int i = tid; i < arr_size; i += num_threads) {
            if (i >= stride) {
                arr[i] += temp[i];
            }
        }
        __syncthreads();
    }
    return;
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_size = blockDim.x;
    __shared__ int sharedHistogram[COMMON_SIZE];
    uchar* curr_in = all_in + block_idx * IMG_HEIGHT * IMG_WIDTH;
    uchar* curr_out = all_out + block_idx * IMG_HEIGHT * IMG_WIDTH;
    uchar* curr_maps = &maps[block_idx * TILE_COUNT * TILE_COUNT * COMMON_SIZE];

    for (int row_tile_n = 0; row_tile_n < TILE_COUNT; row_tile_n++) {
        for (int col_tile_n = 0; col_tile_n < TILE_COUNT; col_tile_n++) {
            // Initialize sharedHistogram/reset everything to zero
            for (int k = tid; k < COMMON_SIZE; k += block_size) {
                sharedHistogram[k] = 0;
            }
            __syncthreads();

            // // Debug print for shared memory initialization
            // if (tid == 0) {
            //     printf("Shared Histogram Initialized:\n");
            //     for (int i = 0; i < COMMON_SIZE; i++) {
            //         printf("%d: %d\t", i, sharedHistogram[i]);
            //     }
            //     printf("\n");
            // }
            // __syncthreads();

            // Fill histogram
            for (int i = tid; i < TILE_WIDTH * TILE_WIDTH; i += block_size) {
                int tile_col = i % TILE_WIDTH;
                int tile_row = i / TILE_WIDTH;
                int y = TILE_WIDTH * row_tile_n + tile_row;
                int x = TILE_WIDTH * col_tile_n + tile_col;
                uchar* row = curr_in + y * IMG_WIDTH;
                atomicAdd(&sharedHistogram[row[x]], 1);
            }
            __syncthreads(); // Ensure all atomic adds are done

            // // Debug print for histogram values
            // if (tid == 0) {
            //     printf("Shared Histogram Filled:\n");
            //     for (int i = 0; i < COMMON_SIZE; i++) {
            //         printf("%d: %d\t", i, sharedHistogram[i]);
            //     }
            //     printf("\n");
            // }
            // __syncthreads();

            // Prefix sum on sharedHistogram
            prefix_sum(sharedHistogram, COMMON_SIZE);
            __syncthreads(); // Ensure prefix sum is completed

            // // Debug print for prefix sum values
            // if (tid == 0) {
            //     printf("Shared Histogram After Prefix Sum:\n");
            //     for (int i = 0; i < COMMON_SIZE; i++) {
            //         printf("%d: %d\t", i, sharedHistogram[i]);
            //     }
            //     printf("\n");
            // }
            // __syncthreads();

            // Get correct maps entry
            uchar* map = &curr_maps[row_tile_n * TILE_COUNT * COMMON_SIZE + col_tile_n * COMMON_SIZE];
            // Create new map values
            for (int k = tid; k < COMMON_SIZE; k += block_size) {
                map[k] = (float(sharedHistogram[k]) * 255) / (TILE_WIDTH * TILE_WIDTH);
            }
            __syncthreads();
        }
    }

    __syncthreads();
    interpolate_device(curr_maps, curr_in, curr_out);
    return;
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
    // task_serial_context needs to hold the pointer to the in_img and out_img and maps
    uchar* maps;
    uchar* in_image;
    uchar* out_image;
    
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;
    //TODO: allocate GPU memory for a single input image, a single output image, and maps
    CUDA_CHECK( cudaMalloc((void**) &(context->maps), sizeof(uchar)*TILE_COUNT*TILE_COUNT*256) );
    CUDA_CHECK( cudaMalloc((void**) &(context->in_image), sizeof(uchar)*IMG_HEIGHT*IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc((void**) &(context->out_image), sizeof(uchar)*IMG_HEIGHT*IMG_WIDTH) );

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, 
                         uchar *images_out)
{
    //TODO: in a for loop:
    uchar* d_maps = context->maps;
    uchar* d_in_image = context->in_image;
    uchar* d_out_image = context->out_image;
    int threads_num = min(1024, THREADS_NUM);
    int sharedMemSize = sizeof(int) * COMMON_SIZE;
    printf("number of threads in serial: %d\n", threads_num);
    dim3 threads_in_block(threads_num), blocks(1);
    for (int i = 0; i < N_IMAGES; i++){
        //   1. copy the relevant image from images_in to the GPU memory you allocated
        CUDA_CHECK( cudaMemcpy(d_in_image, images_in + i*IMG_HEIGHT*IMG_WIDTH, 
                    IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyHostToDevice) );
        //   2. invoke GPU kernel on this image
        process_image_kernel<<<blocks, threads_in_block, sharedMemSize>>>(d_in_image, d_out_image, d_maps);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess){
            fprintf(stderr, "Kernel execution failed:%s\n", cudaGetErrorString(error));
            return;
        }
        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        CUDA_CHECK( cudaMemcpy(images_out + i*IMG_HEIGHT*IMG_WIDTH, d_out_image, 
                    IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyDeviceToHost) );
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    CUDA_CHECK( cudaFree(context->maps) );
    CUDA_CHECK( cudaFree(context->in_image) );
    CUDA_CHECK( cudaFree(context->out_image) );
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
    uchar* all_maps;
    uchar* all_in_images;
    uchar* all_out_images;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps
    CUDA_CHECK( cudaMalloc((void**) &(context->all_maps), sizeof(uchar)*N_IMAGES*TILE_COUNT*TILE_COUNT*256) );
    CUDA_CHECK( cudaMalloc((void**) &(context->all_in_images), sizeof(uchar)*N_IMAGES*IMG_HEIGHT*IMG_WIDTH) );
    CUDA_CHECK( cudaMalloc((void**) &(context->all_out_images), sizeof(uchar)*N_IMAGES*IMG_HEIGHT*IMG_WIDTH) );

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    CUDA_CHECK( cudaMemcpy(context->all_in_images, images_in, 
                N_IMAGES*IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyHostToDevice) );

    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image  
    int threads_num = min(1024, THREADS_NUM);  
    int sharedMemSize = sizeof(int) * COMMON_SIZE;
    dim3 threads_in_block(threads_num), blocks(N_IMAGES);
    process_image_kernel<<<blocks, threads_in_block, sharedMemSize>>>(context->all_in_images,
                                                                      context->all_out_images,
                                                                      context->all_maps);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Kernel execution failed:%s\n", cudaGetErrorString(error));
        return;
    }
    //TODO: copy output images from GPU memory to images_out
    CUDA_CHECK( cudaMemcpy(images_out, context->all_out_images, 
                N_IMAGES*IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyDeviceToHost) );
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init
    CUDA_CHECK( cudaFree(context->all_maps) );
    CUDA_CHECK( cudaFree(context->all_in_images) );
    CUDA_CHECK( cudaFree(context->all_out_images) );
    free(context);
}
