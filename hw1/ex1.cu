#include "ex1.h"

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if (tid >= stride){
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
    return; // TODO
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
    // TODO
    // one thread for each tile
    int block_idx = blockDim.x;
    int tile_col = threadIdx.x;
    int tile_row = threadIdx.y;
        // make it so it would work with less than TILES_COUNT*TILES_COUNT threads for an image
        unsigned int histogram[256] = { 0 };
        int left = TILE_WIDTH*tile_col;
        int right = TILE_WIDTH*(tile_col+1) - 1;
        int top = TILE_WIDTH*tile_row;
        int bottom = TILE_WIDTH*(tile_row+1) - 1;

        // need to do atomic add so when other threads access the same hist
        // they wouldnt read and update the wrong value
        for (int y=top; y<=bottom; y++) {
            for (int x=left; x<=right; x++) {
                uchar* row = all_in + block_idx*IMG_HEIGHT*IMG_WIDTH + y*IMG_WIDTH;
                atomicAdd(&histogram[row[x]], 1);
            }
        }

        int cdf[256] = {0};
        int hist_sum = 0;
        // switch with prefix sum?
        for (int k = 0; k < 256; k++){
            hist_sum += histogram[k];
            cdf[k] = hist_sum;
        }

        uchar* map = &maps[block_idx*TILE_COUNT*TILE_COUNT + tile_row*TILE_COUNT + tile_col];
        for (int k = 0; k < 256; k++){
            map[k] = (float(cdf[k]) * 255) / (TILE_WIDTH*TILE_WIDTH);
        } 
    __syncthreads();
    if (tile_col == 0 && tile_row == 0){
        interpolate_device(maps, all_in + block_idx*IMG_HEIGHT*IMG_WIDTH, all_out + block_idx*IMG_HEIGHT*IMG_WIDTH);
    }
    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
    // task_serial_context needs to hold the pointer to the in_img and out_img and maps
    uchar* maps;
    uchar* in_image;
    uchar* out_image;
    // we have TILES_COUNT*TILES_COUNT tiles
    
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;
    //TODO: allocate GPU memory for a single input image, a single output image, and maps
    cudaMalloc((void**) &(context->maps), sizeof(uchar)*TILE_COUNT*TILE_COUNT*256);
    cudaMalloc((void**) &(context->in_image), sizeof(uchar)*IMG_HEIGHT*IMG_WIDTH);
    cudaMalloc((void**) &(context->out_image), sizeof(uchar)*IMG_HEIGHT*IMG_WIDTH);

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
    dim3 threads_in_block(TILE_COUNT, TILE_COUNT), blocks(1);
    for (int i = 0; i < N_IMAGES; i++){
        //   1. copy the relevant image from images_in to the GPU memory you allocated
        cudaMemcpy(d_in_image, images_in + i*IMG_HEIGHT*IMG_WIDTH, 
                    IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyHostToDevice);
        //   2. invoke GPU kernel on this image
        process_image_kernel<<<blocks, threads_in_block>>>(d_in_image, d_out_image, d_maps);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess){
            fprintf(stderr, "Kernel execution failed:%s\n", cudaGetErrorString(error));
            return;
        }
        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        cudaMemcpy(images_out + i*IMG_HEIGHT*IMG_WIDTH, d_out_image, 
                    IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyDeviceToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    cudaFree(context->maps);
    cudaFree(context->in_image);
    cudaFree(context->out_image);
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
    cudaMalloc((void**) &(context->all_maps), sizeof(uchar)*N_IMAGES*TILE_COUNT*TILE_COUNT*256);
    cudaMalloc((void**) &(context->all_in_images), sizeof(uchar)*N_IMAGES*IMG_HEIGHT*IMG_WIDTH);
    cudaMalloc((void**) &(context->all_out_images), sizeof(uchar)*N_IMAGES*IMG_HEIGHT*IMG_WIDTH);

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    cudaMemcpy(context->all_in_images, images_in, 
                N_IMAGES*IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyHostToDevice);

    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image    
    dim3 threads_in_block(TILE_COUNT, TILE_COUNT), blocks(N_IMAGES);
    process_image_kernel<<<blocks, threads_in_block>>>(context->all_in_images,
                                                        context->all_out_images,
                                                        context->all_maps);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Kernel execution failed:%s\n", cudaGetErrorString(error));
        return;
    }
    //TODO: copy output images from GPU memory to images_out
    cudaMemcpy(images_out, context->all_out_images, 
                N_IMAGES*IMG_HEIGHT*IMG_WIDTH*sizeof(uchar), cudaMemcpyDeviceToHost);
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init
    cudaFree(context->all_maps);
    cudaFree(context->all_in_images);
    cudaFree(context->all_out_images);
    free(context);
}
