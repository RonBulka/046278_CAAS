#include "ex2.h"
#include <cuda/atomic>

#define STREAM_THREADS 1024
#define COMMON_SIZE 256
#define INTERPOLATE_MEM 1024
#define REGS_PER_THREAD 32

class queue_server;
class streams_server;
__global__ void persistent_kernel(queue_server *server);
__device__ void debug_msg(const char* msg, int hist[], int hist_size);
int calculate_threadblocks_count(int threads);
int calculate_upper_log2(int n);

// element of data structure
typedef struct data_element_t {
    int img_id;
    uchar *img_in;
    uchar *img_out;
} data_element;

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
    int num_threads = blockDim.x;
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
                atomicAdd(&sharedHistogram[row[x]], 1);
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

    __syncthreads();
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
                process_image_kernel<<<1, STREAM_THREADS, 0, streams[i]>>>( in_img_d,
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

// TODO implement a Test and Test and Set lock
class TATASLock
{
private:
    // TODO define lock context
    cuda::atomic<int, cuda::thread_scope_device> gpu_lock_state;
public:
    __host__ __device__ TATASLock() : gpu_lock_state(0) {}

    // GPU lock
    __device__ void lock() {
        while (true) {
            // First check if the lock appears to be free
            if (gpu_lock_state.load(cuda::memory_order_acquire) == 0) {
                // Attempt to acquire the lock
                if (gpu_lock_state.exchange(1, cuda::memory_order_acquire) == 0) {
                    return; // Successfully acquired the lock
                }
            }
            // Spin-wait (busy-wait) until the lock is free
        }
    }

    // GPU unlock
    __device__ void unlock() {
        gpu_lock_state.store(0, cuda::memory_order_release);
    }
};



// TODO change things up to work (mainly head and tail stuff)
template <typename T>
class MPMCqueue
{
private:
    size_t size;
    T* queue;
    cuda::atomic<size_t> _head = 0, _tail = 0;
    TATASLock lock;

public:
    __host__ __device__ MPMCqueue() : size(0) {
        this->queue = NULL;
    }

    __host__ __device__ MPMCqueue(size_t N) : size(N) {
        CUDA_CHECK(cudaMallocHost(&(this->queue), sizeof(T) * this->size, 0));
    }

    __host__ __device__ ~MPMCqueue() {
        if (this->queue != NULL) {
            CUDA_CHECK(cudaFreeHost(this->queue));
        }
    }

    __host__ __device__ void init_queue(size_t N) {
        this->size = N;
        CUDA_CHECK(cudaMallocHost(&(this->queue), sizeof(T) * this->size, 0));
    }

    __device__ void gpu_push(const T &data) {
        lock.lock();
        int tail = _tail.load(cuda::memory_order_relaxed);
        while (tail - _head.load(cuda::memory_order_acquire) == size)
            ; // Busy-wait until space is available
        queue[tail % size] = data;
        _tail.store(tail + 1, cuda::memory_order_release);
        lock.unlock();
    }

    __device__ T gpu_pop() {
        lock.lock();
        int head = _head.load(cuda::memory_order_relaxed);
        while (_tail.load(cuda::memory_order_acquire) == _head)
            ; // Busy-wait until data is available
        T item = queue[head % size];
        _head.store(head + 1, cuda::memory_order_release);
        lock.unlock();
        return item;
    }

    __device__ bool is_empty_gpu() {
        lock.lock();
        bool status = _head.load(cuda::memory_order_relaxed) == _tail.load(cuda::memory_order_relaxed);
        lock.unlock();
        return status;
    }

    __host__ void cpu_push(const T &data) {
        int tail = _tail.load(cuda::memory_order_relaxed);
        while (tail - _head.load(cuda::memory_order_acquire) == size)
            ; // Busy-wait until space is available
        queue[tail % size] = data;
        _tail.store(tail + 1, cuda::memory_order_release);
    }

    __host__ T cpu_pop() {
        int head = _head.load(cuda::memory_order_relaxed);
        while (_tail.load(cuda::memory_order_acquire) == _head)
            ; // Busy-wait until data is available
        T item = queue[head % size];
        _head.store(head + 1, cuda::memory_order_release);
        return item;
    }

    __host__ bool is_empty_cpu() {
        return (_head.load(cuda::memory_order_relaxed) == _tail.load(cuda::memory_order_relaxed));
    }

    __host__ bool is_full_cpu() {
        return (_tail.load(cuda::memory_order_relaxed) - _head.load(cuda::memory_order_acquire) == size);
    }
};

// TODO implement the persistent kernel
__global__
void persistent_kernel(queue_server *server) {
    // TODO implement the persistent kernel
    data_element task;
    uchar* block_maps = server->get_maps() + blockIdx.x * TILE_COUNT * TILE_COUNT * COMMON_SIZE;
    while (true) {
        if (server->is_empty_tasks() && server->stop()) {
            break;
        }
        server->gpu_dequeue(&task);
        process_image_kernel(task.img_in, task.img_out, block_maps);
        server->gpu_enqueue(task.img_id);
    }

}

// TODO implement a function for calculating the threadblocks count
int calculate_threadblocks_count(int threads) {
    // set device
    int device;
    cudaDeviceProp deviceProp;
    CUDA_CHECK( cudaGetDevice(&device) );
    CUDA_CHECK( cudaGetDeviceProperties(&deviceProp, device) );
    // get device properties
    int SM_count = deviceProp.multiProcessorCount;
    int max_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    int max_blocks_per_SM = deviceProp.maxBlocksPerMultiProcessor;
    int max_shared_mem_per_SM = deviceProp.sharedMemPerMultiprocessor;
    int max_regs_per_SM = deviceProp.regsPerMultiprocessor;
    // kernel properties
    int threads_per_block = threads;
    int shared_mem_per_block = INTERPOLATE_MEM + 4 * COMMON_SIZE;
    int regs_per_thread = REGS_PER_THREAD;
    // calculate threadblocks count
    // init with max possible threadblocks count
    int threadblocks = max_blocks_per_SM;
    // check threads constraint per SM
    threadblocks = min(threadblocks, (max_threads_per_SM / threads_per_block));
    // check shared memory constraint per SM
    threadblocks = min(threadblocks, (max_shared_mem_per_SM / shared_mem_per_block));
    // check register constraint per SM
    threadblocks = min(threadblocks, (max_regs_per_SM / (threads_per_block * regs_per_thread)));
    // return threadblocks count per all SMs
    return threadblocks * SM_count;
}

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    int thread_blocks;
    int max_queue_size;
    // queue for tasks - cpu pushes and gpu pops
    MPMCqueue<data_element>* tasks;
    // queue for results - gpu pushes and cpu pops
    MPMCqueue<int>* results;
    // flag to stop the kernel
    cuda::atomic<bool> stop_kernel;
    // maps for interpolation
    uchar* taskmaps;
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        stop_kernel.store(false, cuda::memory_order_relaxed);
        this->thread_blocks = calculate_threadblocks_count(threads);
        // calculate max queue size - upper power of 2 of 16 * thread_blocks
        this->max_queue_size = 1 << calculate_upper_log2(this->thread_blocks << 4);
        // init a queue for tasks and a queue for results
        this->tasks = new MPMCqueue<data_element>;
        this->tasks->init_queue(this->max_queue_size);
        this->results = new MPMCqueue<int>;
        this->results->init_queue(this->max_queue_size);
        CUDA_CHECK( cudaMallocHost(&taskmaps, sizeof(uchar) * this->thread_blocks * TILE_COUNT * TILE_COUNT * COMMON_SIZE, 0) );
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        persistent_kernel<<<this->thread_blocks, threads>>>(this);
        
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
        // set stop flag
        stop_kernel.store(true, cuda::memory_order_relaxed);
        // wait for kernel to finish
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess){
            fprintf(stderr, "Kernel execution failed:%s\n", cudaGetErrorString(error));
            return;
        }
        delete this->tasks;
        delete this->results;
        CUDA_CHECK( cudaFreeHost(this->taskmaps) );
    }

    uchar* get_maps() {
        return this->taskmaps;
    }

    // cpu pushes task to task queue
    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        if (this->tasks->is_full_cpu()) {
            return false;
        }
        data_element task = {img_id, img_in, img_out};
        this->tasks->cpu_push(task);
        return true;
    }

    // gpu pops task from queue, processes it, and pushes result to results queue
    __device__ void gpu_dequeue(data_element* task) {
        // TODO pop task from queue
        *task = this->tasks->gpu_pop();
    }

    __device__ void gpu_enqueue(int img_id) {
        // TODO push result into results queue
        this->results->gpu_push(img_id);
    }

    // cpu pops result from results queue
    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        if (this->results->is_empty_cpu()) {
            return false;
        }
        // TODO return the img_id of the request that was completed.
        *img_id = this->results->cpu_pop();
        return true;
    }

    __device__ bool stop() {
        return this->stop_kernel.load(cuda::memory_order_relaxed);
    }

    __device__ bool is_empty_tasks() {
        return this->tasks->is_empty_gpu();
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}