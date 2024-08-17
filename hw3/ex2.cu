/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include "ex2.h"
#include <cuda/atomic>

#define COMMON_SIZE 256
#define REGS_PER_THREAD 32

class queue_server;
class MPMCqueue;

typedef struct data_element_t {
    int img_id;
    uchar *img_in;
    uchar *img_out;
} data_element;

__global__ void persistent_kernel(uchar* maps, MPMCqueue* tasks, MPMCqueue* results, cuda::atomic<bool>* stop_kernel);
__device__ void debug_msg(const char* msg, int hist[], int hist_size);
int calculate_threadblocks_count(int threads);
int calculate_upper_log2(int n);

int calculate_upper_log2(int n) {
    int log2 = 0;
    int flag = 0;
    while (n > 1) {
        if (n % 2 == 1) {
            flag = 1;
        }
        n = n >> 1;
        log2++;
    }
    return log2 + flag;
}

// __device__ void prefix_sum(int arr[], int arr_size) {
//     int tid = threadIdx.x;
//     int num_threads = blockDim.x;
//     extern __shared__ int temp[];
//     for (int stride = 1; stride < arr_size; stride *= 2) {
//         for (int i = tid; i < arr_size; i += num_threads) {
//             if (i >= stride) {
//                 temp[i] = arr[i - stride];
//             }
//         }
//         __syncthreads();
//         for (int i = tid; i < arr_size; i += num_threads) {
//             if (i >= stride) {
//                 arr[i] += temp[i];
//             }
//         }
//         __syncthreads();
//     }
//     return;
// }
__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increase;
    for (int stride = 1; stride < arr_size; stride *= 2) {
        if ((tid >= stride) && (tid < arr_size)) {
            increase = arr[tid - stride];
        } 
        __syncthreads();

        if ((tid >= stride) && (tid < arr_size)) {
            arr[tid] += increase;
        }
        __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tilesâ€™ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);


__device__ void process_image(uchar *all_in, uchar *all_out, uchar* maps) {
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    __shared__ int sharedHistogram[COMMON_SIZE];
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
            //     debug_msg("Shared Histogram Initialized:", sharedHistogram, COMMON_SIZE);
            // }
            // __syncthreads();

            // Fill histogram
            for (int i = tid; i < TILE_WIDTH * TILE_WIDTH; i += block_size) {
                int tile_col = i % TILE_WIDTH;
                int tile_row = i / TILE_WIDTH;
                int y = TILE_WIDTH * row_tile_n + tile_row;
                int x = TILE_WIDTH * col_tile_n + tile_col;
                uchar* row = all_in + y * IMG_WIDTH;
                int pixel = (int)row[x];
                atomicAdd(&sharedHistogram[pixel], 1);
            }
            __syncthreads(); // Ensure all atomic adds are done

            // // Debug print for histogram values
            // if (tid == 0) {
            //     debug_msg("Shared Histogram Filled:", sharedHistogram, COMMON_SIZE);
            // }
            // __syncthreads();

            // Prefix sum on sharedHistogram
            prefix_sum(sharedHistogram, COMMON_SIZE);
            __syncthreads(); // Ensure prefix sum is completed

            // // Debug print for prefix sum values
            // if (tid == 0) {
            //     debug_msg("Shared Histogram After Prefix Sum:", sharedHistogram, COMMON_SIZE);
            // }
            // __syncthreads();

            // Get correct maps entry
            uchar* map = &maps[row_tile_n * TILE_COUNT * COMMON_SIZE + col_tile_n * COMMON_SIZE];
            // Create new map values
            for (int k = tid; k < COMMON_SIZE; k += block_size) {
                map[k] = (float(sharedHistogram[k]) * 255) / (TILE_WIDTH * TILE_WIDTH);
            }
            __syncthreads();
        }
    }
    interpolate_device(maps, all_in, all_out);
    return;
}

__device__
void debug_msg(const char* msg, int hist[], int hist_size) {
    printf("%s\n", msg);
    for (int i = 0; i < hist_size; i++) {
        printf("%d: %d\t", i, hist[i]);
    }
    printf("\n");
}


__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar* maps)
{
    process_image(all_in, all_out, maps);
}


// implement a lock
class TATASLock {
private:
    cuda::atomic<int, cuda::thread_scope_device>* gpu_lock_state;

public:
    __host__ TATASLock() {
        // Initialize lock state
        CUDA_CHECK(cudaMalloc(&gpu_lock_state, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
        CUDA_CHECK(cudaMemset(gpu_lock_state, 0, sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    }

    __host__ ~TATASLock() {
        // Free resources allocated in constructor
        CUDA_CHECK(cudaFree(gpu_lock_state));
    }

    __device__ void lock() {
        while (true) {
            if (gpu_lock_state->load(cuda::memory_order_acquire) == 0) {
                // Try to acquire lock
                if (gpu_lock_state->exchange(1, cuda::memory_order_acquire) == 0) {
                    // Lock acquired, return
                    return;
                }
            }
        }
    }

    __device__ void unlock() {
        gpu_lock_state->store(0, cuda::memory_order_release);
    }
};

// implement a MPMC queue - change_mark
class MPMCqueue {
private:
    size_t max_size;
    data_element* queue;
    cuda::atomic<size_t>* _head;
    cuda::atomic<size_t>* _tail;
    TATASLock lock;

public:
    __host__ MPMCqueue(size_t N) : max_size(N) {
        // Allocate memory for queue
        CUDA_CHECK( cudaHostAlloc(&(this->queue), sizeof(data_element) * this->max_size, cudaHostAllocDefault));
        CUDA_CHECK( cudaHostAlloc(&this->_head, sizeof(cuda::atomic<size_t>), cudaHostAllocDefault) );
        ::new(this->_head) cuda::atomic<size_t>(0);
        CUDA_CHECK( cudaHostAlloc(&this->_tail, sizeof(cuda::atomic<size_t>), cudaHostAllocDefault) );
        ::new(this->_tail) cuda::atomic<size_t>(0);
    }

    __host__ ~MPMCqueue() {
        // Free resources allocated in constructor
        if (this->queue != nullptr) {
            CUDA_CHECK(cudaFreeHost(this->queue));
        }
        if (this->_head != nullptr) {
            this->_head->~atomic<size_t>();
            CUDA_CHECK(cudaFreeHost(this->_head));
        }
        if (this->_tail != nullptr) {
            this->_tail->~atomic<size_t>();
            CUDA_CHECK(cudaFreeHost(this->_tail));
        }
    }

    __device__ bool gpu_push(const data_element &item) {
        lock.lock();
        size_t tail = _tail->load(cuda::memory_order_relaxed);
        if (tail - _head->load(cuda::memory_order_acquire) == max_size) {
            lock.unlock();
            return false;
        }
        queue[tail % max_size] = item;
        _tail->store(tail + 1, cuda::memory_order_release);
        lock.unlock();
        return true;
    }

    __device__ bool gpu_pop(data_element *item) {
        lock.lock();
        size_t head = _head->load(cuda::memory_order_relaxed);
        if (_tail->load(cuda::memory_order_acquire) == head) {
            lock.unlock();
            return false;
        }
        *item = queue[head % max_size];
        _head->store(head + 1, cuda::memory_order_release);
        lock.unlock();
        return true;
    }

    __device__ bool is_empty_gpu() {
        return (_head->load(cuda::memory_order_relaxed) == _tail->load(cuda::memory_order_relaxed));
    }

    __host__ bool cpu_push(const data_element &item) {
        size_t tail = _tail->load(cuda::memory_order_relaxed);
        if (tail - _head->load(cuda::memory_order_acquire) == max_size) {
            return false;
        }
        queue[tail % max_size] = item;
        _tail->store(tail + 1, cuda::memory_order_release);
        return true;
    }

    __host__ bool cpu_pop(data_element *item) {
        size_t head = _head->load(cuda::memory_order_relaxed);
        if (_tail->load(cuda::memory_order_acquire) == head) {
            return false;
        }
        *item = queue[head % max_size];
        _head->store(head + 1, cuda::memory_order_release);
        return true;
    }

    __host__ bool is_empty_cpu() {
        return (_head->load(cuda::memory_order_relaxed) == _tail->load(cuda::memory_order_relaxed));
    }

    __host__ bool is_full_cpu() {
        return (_tail->load(cuda::memory_order_relaxed) - _head->load(cuda::memory_order_relaxed) == max_size);
    }
};

// implement the persistent kernel - change_mark
__global__
void persistent_kernel(uchar* maps, MPMCqueue* tasks, MPMCqueue* results, cuda::atomic<bool>* stop_kernel) {
    __shared__ bool flag;
    __shared__ data_element task;
    uchar* block_maps = maps + blockIdx.x * TILE_COUNT * TILE_COUNT * COMMON_SIZE;
    while (true) {
        if (threadIdx.x == 0) {
            flag = stop_kernel->load(cuda::memory_order_seq_cst) && tasks->is_empty_gpu();
        }
        __syncthreads();
        if (flag) {
            break;
        }
        if (threadIdx.x == 0) {
            flag = !tasks->gpu_pop(&task);
        }
        __syncthreads();
        if (flag) {
            continue;
        }
        __syncthreads();
        process_image(task.img_in, task.img_out, block_maps);
        __syncthreads();
        if (threadIdx.x == 0) {
            while(!results->gpu_push(task));
        }
        __syncthreads();
    }
}

// implement a function for calculating the threadblocks count
int calculate_threadblocks_count(int threads) {
    // get device properties
    int device;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
    int SM_count = deviceProp.multiProcessorCount;
    // printf("SM count: %d\n", SM_count);
    int max_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    // printf("Max threads per SM: %d\n", max_threads_per_SM);
    int max_blocks_per_SM = deviceProp.maxBlocksPerMultiProcessor;
    // printf("Max blocks per SM: %d\n", max_blocks_per_SM);
    int max_shared_mem_per_SM = deviceProp.sharedMemPerMultiprocessor;
    // printf("Max shared memory per SM: %d\n", max_shared_mem_per_SM);
    int max_regs_per_SM = deviceProp.regsPerMultiprocessor;
    // printf("Max regs per SM: %d\n", max_regs_per_SM);

    // get block properties
    int threads_per_block = threads;
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, persistent_kernel));
    int shared_mem_per_block = attr.sharedSizeBytes;
    // printf("Shared memory per block: %d\n", shared_mem_per_block);
    int regs_per_thread = REGS_PER_THREAD;

    // calculate threadblocks
    int threadblocks = max_blocks_per_SM;
    // printf("Max blocks per SM: %d\n", max_blocks_per_SM);
    // thread constraint
    threadblocks = min(threadblocks, (max_threads_per_SM / threads_per_block));
    // printf("Max blocks per SM after threads: %d\n", threadblocks);
    // shared memory constraint
    threadblocks = min(threadblocks, (max_shared_mem_per_SM / shared_mem_per_block));
    // printf("Max blocks per SM after shared mem: %d\n", threadblocks);
    // register constraint
    threadblocks = min(threadblocks, (max_regs_per_SM / (threads_per_block * regs_per_thread)));
    // printf("Max blocks per SM after regs: %d\n", threadblocks);

    return (threadblocks * SM_count);
}


//TODO complete according to HW2
class queue_server : public image_processing_server
{
private:
    int thread_blocks;
    int max_queue_size;
    uchar* pinned_queues;
    MPMCqueue* tasks;
    MPMCqueue* results;
    cuda::atomic<bool>* stop_kernel;
    uchar* taskmaps;

public:
    queue_server(int threads) {
        // Allocate memory for atomic<bool>
        CUDA_CHECK( cudaHostAlloc(&this->stop_kernel, sizeof(cuda::atomic<bool>), cudaHostAllocDefault) );
        ::new(this->stop_kernel) cuda::atomic<bool>(false);

        // Calculate threadblocks count and max queue size
        thread_blocks = calculate_threadblocks_count(threads);
        max_queue_size = 1 << calculate_upper_log2(thread_blocks << 4);

        // Allocate memory for queues
        CUDA_CHECK(cudaMallocHost(&pinned_queues, sizeof(MPMCqueue) * 2));
        tasks   = new (pinned_queues) MPMCqueue(max_queue_size);
        results = new (pinned_queues + sizeof(MPMCqueue)) MPMCqueue(max_queue_size);
        CUDA_CHECK(cudaMalloc(&taskmaps, thread_blocks * TILE_COUNT * TILE_COUNT * COMMON_SIZE * sizeof(uchar)));

        dim3 _threads(threads), blocks(thread_blocks);
        persistent_kernel<<<blocks, _threads>>>(taskmaps, tasks, results, stop_kernel);
    }

    ~queue_server() override {
        // Send signal to stop kernel and wait for it to finish
        stop_kernel->store(true, cuda::memory_order_seq_cst);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(error));
            return;
        }
        // Free resources allocated in constructor
        if (stop_kernel != nullptr) {
            stop_kernel->~atomic<bool>();
            CUDA_CHECK(cudaFreeHost(stop_kernel));
        }
        if (tasks != nullptr) {
            this->tasks->~MPMCqueue();
        }
        if (results != nullptr) {
            this->results->~MPMCqueue();
        }
        if (pinned_queues != nullptr) {
            CUDA_CHECK(cudaFreeHost(pinned_queues));
        }
        CUDA_CHECK(cudaFree(taskmaps));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override {
        //TODO complete according to HW2 - change_mark
        data_element task = {};
        task.img_id = img_id;
        task.img_in = img_in;
        task.img_out = img_out;
        return tasks->cpu_push(task);
    }

    bool dequeue(int *img_id) override
    {
        //TODO complete according to HW2 - change_mark
        data_element task;
        if (!results->cpu_pop(&task)) {
            return false;
        }
        *img_id = task.img_id;
        return true;
    }
};


std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
