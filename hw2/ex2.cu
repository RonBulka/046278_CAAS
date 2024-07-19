#include "ex2.h"
#include <cuda/atomic>

#define STREAM_THREADS 1024
#define COMMON_SIZE 256
#define INTERPOLATE_MEM 1024
#define REGS_PER_THREAD 32

class queue_server;
class streams_server;
template <typename T> class MPMCqueue;

// element of data structure
typedef struct data_element_t {
    int img_id;
    uchar *img_in;
    uchar *img_out;
} data_element;

__global__ void persistent_kernel(uchar* maps, MPMCqueue<data_element>* tasks,
                                  MPMCqueue<int>* results, cuda::atomic<bool>* stop_kernel);
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
                // printf("Trying to acquire lock\n");
                if (gpu_lock_state.exchange(1, cuda::memory_order_acquire) == 0) {
                    printf("Lock acquired\n");
                    return; // Successfully acquired the lock
                }
            }
            // Spin-wait (busy-wait) until the lock is free
        }
    }

    // GPU unlock
    __device__ void unlock() {
        gpu_lock_state.store(0, cuda::memory_order_release);
        printf("Lock released\n");
    }
};

// TODO change things up to work (mainly head and tail stuff)
template <typename T>
class MPMCqueue
{
private:
    size_t max_size;
    T* queue;
    cuda::atomic<size_t> _head, _tail, _size;
    TATASLock lock;

public:
    __host__ MPMCqueue() : max_size(0), queue(nullptr) {
        _head.store(0, cuda::memory_order_seq_cst);
        _tail.store(0, cuda::memory_order_seq_cst);
        _size.store(0, cuda::memory_order_seq_cst);
    }

    __host__ MPMCqueue(size_t N) : max_size(N) {
        CUDA_CHECK(cudaHostAlloc(&(this->queue), sizeof(T) * this->max_size, cudaHostAllocDefault));
        _head.store(0, cuda::memory_order_seq_cst);
        _tail.store(0, cuda::memory_order_seq_cst);
        _size.store(0, cuda::memory_order_seq_cst);
    }

    __host__ ~MPMCqueue() {
        if (this->queue != nullptr) {
            CUDA_CHECK(cudaFreeHost(this->queue));
        }
    }

    __host__ void init_queue(size_t N) {
        this->max_size = N;
        CUDA_CHECK(cudaHostAlloc(&(this->queue), sizeof(T) * this->max_size, cudaHostAllocDefault));
    }

    __device__ bool gpu_push(const T item) {
        lock.lock();
        if (_size.load(cuda::memory_order_acquire) == max_size) {
            // Queue is full
            _size.store(max_size, cuda::memory_order_release);
            lock.unlock();
            return false;
        }
        int tail = _tail.load(cuda::memory_order_seq_cst);
        queue[tail % max_size] = item;
        _tail.store((tail + 1) % max_size, cuda::memory_order_seq_cst);
        _size.fetch_add(1, cuda::memory_order_release); // Increment size
        lock.unlock();
        return true;
    }

    __device__ bool gpu_pop(T *item) {
        lock.lock();
        if (_size.load(cuda::memory_order_acquire) == 0) {
            // Queue is empty
            _size.store(0, cuda::memory_order_release);
            lock.unlock();
            return false;
        }
        int head = _head.load(cuda::memory_order_seq_cst);
        *item = queue[head % max_size];
        _head.store((head + 1) % max_size, cuda::memory_order_seq_cst);
        _size.fetch_sub(1, cuda::memory_order_release); // Decrement size
        lock.unlock();
        return true;
    }

    __device__ bool is_empty_gpu() {
        lock.lock();
        bool status = (_size.load(cuda::memory_order_seq_cst) == 0);
        lock.unlock();
        return status;
    }

    __host__ bool cpu_push(const T item) {
        if (_size.load(cuda::memory_order_acquire) == max_size) {
            // Queue is full
            _size.store(max_size, cuda::memory_order_release);
            return false;
        }
        int tail = _tail.load(cuda::memory_order_seq_cst);
        queue[tail % max_size] = item;
        _tail.store((tail + 1) % max_size, cuda::memory_order_seq_cst);
        _size.fetch_add(1, cuda::memory_order_release); // Increment size
        return true;
    }

    __host__ bool cpu_pop(T *item) {
        if (_size.load(cuda::memory_order_acquire) == 0) {
            // Queue is empty
            _size.store(0, cuda::memory_order_release);
            return false;
        }
        int head = _head.load(cuda::memory_order_seq_cst);
        *item = queue[head % max_size];
        _head.store((head + 1) % max_size, cuda::memory_order_seq_cst);
        _size.fetch_sub(1, cuda::memory_order_release); // Decrement size
        return true;
    }

    __host__ bool is_empty_cpu() {
        return (_size.load(cuda::memory_order_seq_cst) == 0);
    }

    __host__ bool is_full_cpu() {
        return (_size.load(cuda::memory_order_seq_cst) == max_size);
    }
};



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
    int shared_mem_per_block = INTERPOLATE_MEM + sizeof(int) * COMMON_SIZE + sizeof(bool);
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

// TODO implement the persistent kernel
__global__
void persistent_kernel(uchar* maps, MPMCqueue<data_element>* tasks, 
                       MPMCqueue<int>* results, cuda::atomic<bool>* stop_kernel) {
    if (threadIdx.x == 0) {
        printf("Thread block %d is alive\n", blockIdx.x);
    }
    __shared__ bool flag;
    data_element task;
    uchar* block_maps = maps + blockIdx.x * TILE_COUNT * TILE_COUNT * COMMON_SIZE;
    while (true) {
        if (threadIdx.x == 0) {
            printf("Thread block %d entered loop\n", blockIdx.x);
            flag = false;
            if (stop_kernel->load(cuda::memory_order_seq_cst) && tasks->is_empty_gpu()) {
                printf("Thread block %d is done\n", blockIdx.x);
                flag = true;
            }
        }
        __syncthreads();
        if (flag) {
            break;
        }
        if (threadIdx.x == 0) {
            printf("Thread block %d is trying to dequeue\n", blockIdx.x);
            if (!tasks->gpu_pop(&task)) {
                flag = true;
            }
        }
        __syncthreads();
        if (flag) {
            continue;
        }
        process_image(task.img_in, task.img_out, block_maps);
        if (threadIdx.x == 0) {
            while (!results->gpu_push(task.img_id));
        }
        __syncthreads();
    }
}

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    int thread_blocks;
    int max_queue_size;
    // queue for tasks - cpu pushes and gpu pops
    uchar* pinned_queue_tasks;
    MPMCqueue<data_element>* tasks;
    // queue for results - gpu pushes and cpu pops
    uchar* pinned_queue_results;
    MPMCqueue<int>* results;
    // flag to stop the kernel
    cuda::atomic<bool> stop_kernel;
    // maps for interpolation
    uchar* taskmaps;
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        stop_kernel.store(false, cuda::memory_order_seq_cst);
        this->thread_blocks = calculate_threadblocks_count(threads);
        printf("Thread blocks: %d\n", this->thread_blocks);
        // calculate max queue size - upper power of 2 of 16 * thread_blocks
        this->max_queue_size = 1 << calculate_upper_log2(this->thread_blocks << 4);
        printf("Max queue size: %d\n", this->max_queue_size);
        // init a queue for tasks and a queue for results
        CUDA_CHECK( cudaHostAlloc(&pinned_queue_tasks, sizeof(MPMCqueue<data_element>), cudaHostAllocDefault) );
        this->tasks = new (pinned_queue_tasks) MPMCqueue<data_element>;
        this->tasks->init_queue(this->max_queue_size);

        CUDA_CHECK( cudaHostAlloc(&pinned_queue_results, sizeof(MPMCqueue<int>), cudaHostAllocDefault) );
        this->results = new (pinned_queue_results) MPMCqueue<int>;
        this->results->init_queue(this->max_queue_size);

        CUDA_CHECK( cudaHostAlloc(&taskmaps, sizeof(uchar) * this->thread_blocks * TILE_COUNT * TILE_COUNT * COMMON_SIZE, cudaHostAllocDefault) );
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        dim3 _threads(threads), blocks(this->thread_blocks);
        persistent_kernel<<<blocks, _threads>>>(this->taskmaps, this->tasks,
                                                this->results, &this->stop_kernel);
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
        // set stop flag
        stop_kernel.store(true, cuda::memory_order_seq_cst);
        // wait for kernel to finish
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess){
            fprintf(stderr, "Kernel execution failed:%s\n", cudaGetErrorString(error));
            return;
        }
        delete this->tasks;
        CUDA_CHECK( cudaFreeHost(this->pinned_queue_tasks) );
        delete this->results;
        CUDA_CHECK( cudaFreeHost(this->pinned_queue_results) );
        CUDA_CHECK( cudaFreeHost(this->taskmaps) );
    }

    __device__ uchar* get_maps() {
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
        printf("Image %d is in queue\n", img_id);
        return true;
    }

    __device__ bool gpu_enqueue(int img_id) {
        // TODO push result into results queue
        return this->results->gpu_push(img_id);
    }

    // cpu pops result from results queue
    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        if (this->results->is_empty_cpu()) {
            return false;
        }
        // TODO return the img_id of the request that was completed.
        this->results->cpu_pop(img_id);
        printf("Image %d is ready\n", *img_id);
        return true;
    }

    // gpu pops task from queue
    __device__ bool gpu_dequeue(data_element* task) {
        return this->tasks->gpu_pop(task);
    }

    __device__ bool stop() {
        return this->stop_kernel.load(cuda::memory_order_seq_cst);
    }

    __device__ bool is_empty_tasks() {
        return this->tasks->is_empty_gpu();
    }
};


std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}