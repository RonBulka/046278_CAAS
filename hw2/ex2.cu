#include "ex2.h"
#include <cuda/atomic>
#include <new>

#define STREAM_THREADS 1024
#define COMMON_SIZE 256
#define INTERPOLATE_MEM 1024
#define REGS_PER_THREAD 32

class queue_server;
class streams_server;
class MPMCqueue;

// element of data structure
typedef struct data_element_t {
    int img_id;
    uchar *img_in;
    uchar *img_out;
} data_element;

__global__ void persistent_kernel(uchar* maps, MPMCqueue* tasks,
                                  MPMCqueue* results, cuda::atomic<bool>* stop_kernel);
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

__device__ void prefix_sum(int arr[], int arr_size) {
    // TODO complete according to hw1
    int tid = threadIdx.x;
    // int num_threads = blockDim.x;
    // extern __shared__ int temp[];
    int increase;
    for (int stride = 1; stride < arr_size; stride *= 2) {
        // for (int i = tid; i < arr_size; i += num_threads) {
        //     if (i >= stride) {
        //         temp[i] = arr[i - stride];
        //     }
        // }
        if ((tid >= stride) && (tid < arr_size)) {
            increase = arr[tid - stride];
        } 
        __syncthreads();
        // for (int i = tid; i < arr_size; i += num_threads) {
        //     if (i >= stride) {
        //         arr[i] += temp[i];
        //     }
        // }
        if ((tid >= stride) && (tid < arr_size)) {
            arr[tid] += increase;
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
            //     printf("Thread block %d Processing tile %d %d\n", blockIdx.x, row_tile_n, col_tile_n);
            //     debug_msg("Shared Histogram Initialized:", sharedHistogram, COMMON_SIZE);
            // }
            // __syncthreads();

            // Fill histogram
            for (int i = tid; i < TILE_WIDTH * TILE_WIDTH; i += block_size) {
                int tile_col = i % TILE_WIDTH;
                int tile_row = i / TILE_WIDTH;
                int y = TILE_WIDTH * row_tile_n + tile_row;
                int x = TILE_WIDTH * col_tile_n + tile_col;
                uchar* row = in + y * IMG_WIDTH;
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
    // if (tid == 0) {
    //     printf("Thread block %d before interpolate_device\n", blockIdx.x);
    // }
    // __syncthreads();
    interpolate_device(maps, in, out);
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

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

/*****************************************************************************/
// Streams implemintation
/*****************************************************************************/
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
    streams_server() {
        // TODO initialize context (memory buffers, streams, etc...)
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            image_id[i] = -1;
        }
        CUDA_CHECK(cudaHostAlloc(&maps, sizeof(uchar) * STREAM_COUNT * TILE_COUNT * TILE_COUNT * COMMON_SIZE, cudaHostAllocDefault));
    }

    ~streams_server() override {
        // TODO free resources allocated in constructor
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        CUDA_CHECK(cudaFreeHost(maps));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override {
        // TODO place memory transfers and kernel invocation in streams if possible.
        for (int i = 0; i < STREAM_COUNT; i++) {
            if (image_id[i] == -1) {
                image_id[i] = img_id;
                uchar* maps_d = this->maps + i * TILE_COUNT * TILE_COUNT * COMMON_SIZE;
                process_image_kernel<<<1, STREAM_THREADS, 0, streams[i]>>>(img_in, img_out, maps_d);
                return true;
            }
        }
        return false;
    }

    bool dequeue(int *img_id) override {
        // TODO query (don't block) streams for any completed requests.
        for (int i = 0; i < STREAM_COUNT; i++) {
            cudaError_t status = cudaStreamQuery(streams[i]); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                if (image_id[i] == -1) {
                    continue;
                }
                *img_id = image_id[i];
                // printf("Image %d is ready\n", *img_id);
                image_id[i] = -1;
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

/*****************************************************************************/
// Queue implemintation
/*****************************************************************************/
// TODO implement a Test and Test and Set lock
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
                if (gpu_lock_state->exchange(1, cuda::memory_order_acquire) == 0) {
                    return;
                }
            }
        }
    }

    __device__ void unlock() {
        gpu_lock_state->store(0, cuda::memory_order_release);
    }
};

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
        // printf("Thread block %d pushed image %d\n", blockIdx.x, item.img_id);
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
        // printf("Thread block %d popped image %d with value of %d in last place\n", blockIdx.x, item->img_id, item->img_in[IMG_HEIGHT * IMG_WIDTH - 1]);
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

int calculate_threadblocks_count(int threads) {
    // get device properties
    int device;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
    int SM_count = deviceProp.multiProcessorCount;
    int max_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    int max_blocks_per_SM = deviceProp.maxBlocksPerMultiProcessor;
    int max_shared_mem_per_SM = deviceProp.sharedMemPerMultiprocessor;
    int max_regs_per_SM = deviceProp.regsPerMultiprocessor;

    // get block properties
    int threads_per_block = threads;
    int shared_mem_per_block = INTERPOLATE_MEM + sizeof(int) * COMMON_SIZE + sizeof(data_element) + sizeof(bool);
    int regs_per_thread = REGS_PER_THREAD;

    // calculate threadblocks
    int threadblocks = max_blocks_per_SM;
    // thread constraint
    threadblocks = min(threadblocks, (max_threads_per_SM / threads_per_block));
    // shared memory constraint
    threadblocks = min(threadblocks, (max_shared_mem_per_SM / shared_mem_per_block));
    // register constraint
    threadblocks = min(threadblocks, (max_regs_per_SM / (threads_per_block * regs_per_thread)));

    return threadblocks * SM_count;
}

__global__
void persistent_kernel(uchar* maps, MPMCqueue* tasks, MPMCqueue* results, cuda::atomic<bool>* stop_kernel) {
    // if (threadIdx.x == 0) {
    //     printf("Thread block %d is alive\n", blockIdx.x);
    // }
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
            flag = tasks->gpu_pop(&task);
        }
        __syncthreads();
        if (!flag) {
            continue;
        }
        __syncthreads();
        process_image(task.img_in, task.img_out, block_maps);
        __syncthreads();
        if (threadIdx.x == 0) {
            // printf("Thread block %d finished processing image %d\n", blockIdx.x, task.img_id);
            while(!results->gpu_push(task));
        }
        __syncthreads();
    }
}

class queue_server : public image_processing_server {
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

        thread_blocks = calculate_threadblocks_count(threads);
        // printf("Thread blocks: %d\n", thread_blocks);
        max_queue_size = 1 << calculate_upper_log2(thread_blocks << 4);
        // printf("Max queue size: %d\n", max_queue_size);

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
        // printf("Freeing resources\n");
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

    bool enqueue(int img_id, uchar* img_in, uchar* img_out) override {
        data_element task = {};
        task.img_id = img_id;
        task.img_in = img_in;
        task.img_out = img_out;
        return tasks->cpu_push(task);
    }

    bool dequeue(int* img_id) override {
        data_element task;
        if (!results->cpu_pop(&task)) {
            return false;
        }
        *img_id = task.img_id;
        // printf("Image %d is ready\n", *img_id);
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}