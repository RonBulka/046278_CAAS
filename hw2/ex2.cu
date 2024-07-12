#include "ex2.h"
#include <cuda/atomic>

#define THREADS_NUM TILE_WIDTH*4
#define STREAM_THREADS 1024
#define COMMON_SIZE 256
#define MAX_SHARED_MEM_BLK 49152
#define MAX_REGS_BLK 65536
#define MAX_THREADS_BLK 1024
#define MAX_BLOCKS_PER_SM 16

__device__ void prefix_sum(int arr[], int arr_size) {
    // TODO complete according to hw1
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

__device__
void process_image(uchar *in, uchar *out, uchar* maps) {
    // TODO complete according to hw1
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_size = blockDim.x;
    __shared__ int sharedHistogram[COMMON_SIZE];
    uchar* curr_in = in + block_idx * IMG_HEIGHT * IMG_WIDTH;
    uchar* curr_out = out + block_idx * IMG_HEIGHT * IMG_WIDTH;
    uchar* curr_maps = &maps[block_idx * TILE_COUNT * TILE_COUNT * COMMON_SIZE];
    // do one tile at a time
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

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    cudaStream_t streams[STREAM_COUNT];
    uchar* maps;
    uchar* in_image;
    uchar* out_image;
    int image_id[STREAM_COUNT];

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
        CUDA_CHECK(cudaHostAlloc(&maps, sizeof(uchar) * STREAM_COUNT * TILE_COUNT * TILE_COUNT * COMMON_SIZE, 0));
        CUDA_CHECK(cudaHostAlloc(&in_image, sizeof(uchar) * STREAM_COUNT * IMG_HEIGHT * IMG_WIDTH, 0));
        CUDA_CHECK(cudaHostAlloc(&out_image, sizeof(uchar) * STREAM_COUNT * IMG_HEIGHT * IMG_WIDTH, 0));
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        CUDA_CHECK(cudaFreeHost(maps));
        CUDA_CHECK(cudaFreeHost(in_image));
        CUDA_CHECK(cudaFreeHost(out_image));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            if (image_id[i] == -1)
            {
                image_id[i] = img_id;
                uchar* maps_d = this->maps + i * TILE_COUNT * TILE_COUNT * COMMON_SIZE;
                uchar* in_img_d = this->in_image + i * IMG_HEIGHT * IMG_WIDTH;
                uchar* out_img_d = this->out_image + i * IMG_HEIGHT * IMG_WIDTH;
                CUDA_CHECK(cudaMemcpyAsync(in_img_d, img_in, sizeof(uchar) * IMG_HEIGHT * IMG_WIDTH, 
                                            cudaMemcpyHostToDevice, streams[i]));
                process_image_kernel<<<1, STREAM_THREADS, sizeof(int) * COMMON_SIZE, streams[i]>>>( in_img_d,
                                                                                                    out_img_d, 
                                                                                                    maps_d);
                CUDA_CHECK(cudaMemcpyAsync(img_out, out_img_d, sizeof(uchar) * IMG_HEIGHT * IMG_WIDTH, 
                                            cudaMemcpyDeviceToHost, streams[i]));
                return true;
            }
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        return false;

        // TODO query (don't block) streams for any completed requests.
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            cudaError_t status = cudaStreamQuery(streams[i]); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                *img_id = image_id[i];
                image_id[i] = -1;
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        }
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the img_id of the request that was completed.
        //*img_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
