/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>

#define KILL_SERVER (-1)
#define HEAD (true)
#define TAIL (false)
#define COPY_TO_SERVER (true)
#define COPY_FROM_SERVER (false)

/********************************* RPC implementation *********************************/
class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_src
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_dst
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    while(!gpu_context->enqueue(wc.wr_id, img_in, img_out)){};
		    break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};

/************************************ Helper functions ***********************************/
void post_atomic_fa(struct ibv_qp *qp, uint64_t remote_addr, uint32_t rkey, uint64_t add_val) {
    struct ibv_sge sge;
    sge.addr = 0;               // FA doesn't use local data
    sge.length = 0;
    sge.lkey = 0;

    struct ibv_send_wr wr = {};
    wr.wr_id = 0;               // User-defined ID
    wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey = rkey;
    wr.wr.atomic.compare_add = add_val;

    struct ibv_send_wr *bad_wr = nullptr;
    int ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
        fprintf(stderr, "Failed to post atomic fetch-and-add operation\n");
        exit(1);
    }
}

void post_atomic_cas(struct ibv_qp *qp, uint64_t remote_addr, uint32_t rkey, uint64_t new_val, 
                        uint64_t expected_val, uint32_t lkey, uint64_t *previous_val) {
    struct ibv_sge sge;
    sge.addr = (uintptr_t)previous_val;     // Local address to store previous value
    sge.length = sizeof(uint64_t);
    sge.lkey = lkey;                        // Local memory region key

    struct ibv_send_wr wr = {};
    wr.wr_id = 0;               // User-defined ID
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey = rkey;
    wr.wr.atomic.compare_add = expected_val;    // This is the compare value in CAS
    wr.wr.atomic.swap = new_val;                // This is the swap value in CAS

    struct ibv_send_wr *bad_wr = nullptr;
    int ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
        fprintf(stderr, "Failed to post atomic compare-and-swap operation\n");
        exit(1);
    }
}

/********************************* Queues implementation *********************************/

class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> server;

    /* TODO: add memory region(s) for CPU-GPU queues */
    struct ibv_mr *mr_tasks_queue;          // Memory region for tasks queue
    struct ibv_mr *mr_tasks_head;           // Memory region for tasks head
    struct ibv_mr *mr_tasks_tail;           // Memory region for tasks tail
    struct ibv_mr *mr_results_queue;        // Memory region for results queue
    struct ibv_mr *mr_results_head;         // Memory region for results head
    struct ibv_mr *mr_results_tail;         // Memory region for results tail

    // helper data
    size_t queue_size;
    MPMCqueue *tasks_MPMCqueue;
    data_element *tasks_queue;
    cuda::atomic<size_t> *tasks_head;
    cuda::atomic<size_t> *tasks_tail;

    MPMCqueue *results_MPMCqueue;
    data_element *results_queue;
    cuda::atomic<size_t> *results_head;
    cuda::atomic<size_t> *results_tail;
public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port) {
        server = create_queues_server(THREADS_PER_BLOCK);
        /* TODO Initialize additional server MRs as needed. */
        enum ibv_access_flags access_flags = static_cast<ibv_access_flags>(IBV_ACCESS_LOCAL_WRITE | \
                                                                           IBV_ACCESS_REMOTE_READ | \
                                                                           IBV_ACCESS_REMOTE_WRITE);
        queue_size = server->get_max_queue_size();
        tasks_MPMCqueue = server->get_tasks();
        tasks_queue = tasks_MPMCqueue->get_queue();
        tasks_head = tasks_MPMCqueue->get__head();
        tasks_tail = tasks_MPMCqueue->get__tail();

        results_MPMCqueue = server->get_results();
        results_queue = results_MPMCqueue->get_queue();
        results_head = results_MPMCqueue->get__head();
        results_tail = results_MPMCqueue->get__tail();

        mr_tasks_queue = ibv_reg_mr(pd, tasks_queue, queue_size * sizeof(data_element), access_flags);
        if (!mr_tasks_queue) {
            perror("ibv_reg_mr() failed for tasks queue");
            exit(1);
        }

        mr_tasks_head = ibv_reg_mr(pd, tasks_head, sizeof(cuda::atomic<size_t>), access_flags | IBV_ACCESS_REMOTE_ATOMIC);
        if (!mr_tasks_head) {
            perror("ibv_reg_mr() failed for tasks head");
            exit(1);
        }

        mr_tasks_tail = ibv_reg_mr(pd, tasks_tail, sizeof(cuda::atomic<size_t>), access_flags | IBV_ACCESS_REMOTE_ATOMIC);
        if (!mr_tasks_tail) {
            perror("ibv_reg_mr() failed for tasks tail");
            exit(1);
        }

        mr_results_queue = ibv_reg_mr(pd, results_queue, queue_size * sizeof(data_element), access_flags);
        if (!mr_results_queue) {
            perror("ibv_reg_mr() failed for results queue");
            exit(1);
        }

        mr_results_head = ibv_reg_mr(pd, results_head, sizeof(cuda::atomic<size_t>), access_flags | IBV_ACCESS_REMOTE_ATOMIC);
        if (!mr_results_head) {
            perror("ibv_reg_mr() failed for results head");
            exit(1);
        }

        mr_results_tail = ibv_reg_mr(pd, results_tail, sizeof(cuda::atomic<size_t>), access_flags | IBV_ACCESS_REMOTE_ATOMIC);
        if (!mr_results_tail) {
            perror("ibv_reg_mr() failed for results tail");
            exit(1);
        }

        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
        // TODO: send the rkeys and addresses to the client
        // init struct to send to client
        rdma_connection_info info = {};
        init_rdma_connection_info(&info);

        /* send the connection info to the client */
        send_over_socket(&info, sizeof(rdma_connection_info));
    }

    ~server_queues_context() {
        /* TODO destroy the additional server MRs here */

        ibv_dereg_mr(this->mr_tasks_queue);
        ibv_dereg_mr(this->mr_tasks_head);
        ibv_dereg_mr(this->mr_tasks_tail);

        ibv_dereg_mr(this->mr_results_queue);
        ibv_dereg_mr(this->mr_results_head);
        ibv_dereg_mr(this->mr_results_tail);

        // server->~queue_server();
    }

    virtual void event_loop() override {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
        rpc_request* req;
        bool terminate = false;

        while (!terminate) {
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);

            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }

            if (ncqes > 0) {
                VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                    case IBV_WC_RECV:
                        /* Received a new request from the client */
                        req = &requests[wc.wr_id];

                        /* Terminate signal */
                        if (req->request_id == KILL_SERVER) {
                            //printf("Terminating...\n");
                            terminate = true;
                            post_rdma_write(
                                req->output_addr,                       // remote_dst
                                0,     // len
                                req->output_rkey,                       // rkey
                                0,                // local_src
                                mr_images_out->lkey,                    // lkey
                                (uint64_t)KILL_SERVER, // wr_id
                                (uint32_t *)&req->request_id);          // immediate
                        } else {
                            printf("Unexpected error\n");
                            assert(false);
                        }
                    break;
                    default:
                        printf("Unexpected completion\n");
                        assert(false);
                }
            }
        }
    }

    void init_rdma_connection_info(rdma_connection_info *info) {
        info->images_in_rkey = this->mr_images_in->rkey;
        info->images_in_addr = (uintptr_t)this->mr_images_in->addr;

        info->images_out_rkey = this->mr_images_out->rkey;
        info->images_out_addr = (uintptr_t)this->mr_images_out->addr;

        info->tasks_queue_rkey = mr_tasks_queue->rkey;
        info->tasks_queue_addr = (uintptr_t)tasks_queue;

        info->tasks_head_rkey = mr_tasks_head->rkey;
        info->tasks_head_addr = (uintptr_t)tasks_head;

        info->tasks_tail_rkey = mr_tasks_tail->rkey;
        info->tasks_tail_addr = (uintptr_t)tasks_tail;

        info->results_queue_rkey = mr_results_queue->rkey;
        info->results_queue_addr = (uintptr_t)results_queue;

        info->results_head_rkey = mr_results_head->rkey;
        info->results_head_addr = (uintptr_t)results_head;

        info->results_tail_rkey = mr_results_tail->rkey;
        info->results_tail_addr = (uintptr_t)results_tail;

        info->queue_size = queue_size;
    }
};

class client_queues_context : public rdma_client_context {
private:
    /* TODO add necessary context to track the client side of the GPU's
     * producer/consumer queues */
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    // Data_elements to send and recieve data
    data_element sending_data = {};
    data_element recieved_data = {};

    uchar* images_out_addr;

    struct ibv_mr *mr_sending_data_element;
    struct ibv_mr *mr_recieved_data_element;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
    /* TODO define other memory regions used by the client here */
    rdma_connection_info remote_info;

    rdma_queues_indexes queues_indexes;
    struct ibv_mr *mr_queues_indexes;

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        recv_over_socket(&remote_info, sizeof(rdma_connection_info));
        init_mr_data_elements();
        init_mr_queues_indexes();
    }

    ~client_queues_context()
    {
        /* terminate the server */
        kill();
        /* release memory regions and other resources */
        ibv_dereg_mr(mr_queues_indexes);
        ibv_dereg_mr(mr_recieved_data_element);
        ibv_dereg_mr(mr_sending_data_element);
        ibv_dereg_mr(mr_images_in);
        ibv_dereg_mr(mr_images_out);
    }

    void init_mr_data_elements() {
        mr_sending_data_element = ibv_reg_mr(pd, &sending_data, sizeof(data_element), IBV_ACCESS_LOCAL_WRITE);
        if (!mr_sending_data_element) {
            perror("ibv_reg_mr() failed for sending data element");
            exit(1);
        }
        mr_recieved_data_element = ibv_reg_mr(pd, &recieved_data, sizeof(data_element), IBV_ACCESS_LOCAL_WRITE);
        if (!mr_recieved_data_element) {
            perror("ibv_reg_mr() failed for recieved data element");
            exit(1);
        }
    }

    void init_mr_queues_indexes() {
        queues_indexes.tasks_head = 0;
        queues_indexes.tasks_tail = 0;
        queues_indexes.results_head = 0;
        queues_indexes.results_tail = 0;
        mr_queues_indexes = ibv_reg_mr(pd, &queues_indexes, sizeof(rdma_queues_indexes), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_queues_indexes) {
            perror("ibv_reg_mr() failed for queues indexes");
            exit(1);
        }
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        // TODO register memory
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        // TODO register memory
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
        images_out_addr = images_out;
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */

        // read _head from remote tasks
        read_index(HEAD);

        // check if full
        if (queues_indexes.tasks_tail - queues_indexes.tasks_head == remote_info.queue_size) {
            return false;
        }

        // copy in image to server
        uchar *remote_image_in = (uchar *)remote_info.images_in_addr;
        uchar *image_in_dst = &remote_image_in[(img_id % OUTSTANDING_REQUESTS) * IMG_SZ];
        copy_image(queues_indexes.tasks_tail, img_in, image_in_dst, COPY_TO_SERVER);

        uchar *remote_image_out = (uchar *)remote_info.images_out_addr;
        uchar *image_out_dst = &remote_image_out[(img_id % OUTSTANDING_REQUESTS) * IMG_SZ];

        // create matching data element
        sending_data = {img_id, image_in_dst, image_out_dst};

        // insert data element in task queue
        enqueue_data_element(queues_indexes.tasks_tail, &sending_data);

        // update tasks _tail index
        update_index(TAIL);

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */

        // read _tail from remote results
        read_index(TAIL);

        // check if empty
        if (queues_indexes.results_tail == queues_indexes.results_head) {
            return false;
        }

        // read data element from results queue
        dequeue_data_element(queues_indexes.results_head, &recieved_data);

        // copy out image to client
        uchar *local_image_out = &images_out_addr[(recieved_data.img_id % N_IMAGES) * IMG_SZ];
        copy_image(queues_indexes.results_head, local_image_out, recieved_data.img_out, COPY_FROM_SERVER);

        // set image id
        *img_id = recieved_data.img_id;

        // update results _head index
        update_index(HEAD);

        return true;
    }

// go over the functions really carefully
    void read_index(bool flag) {
        // can stay normal reads
        void *local_dst = NULL;
        uint64_t remote_src = 0;    
        uint32_t rkey = 0;    
        uint64_t wr_id = 0;
        int ncqes = 0;

        if (flag == TAIL) {
            local_dst = &queues_indexes.results_tail;
            remote_src = remote_info.results_tail_addr;     // remote_src
            rkey = remote_info.results_tail_rkey;           // rkey
            wr_id = queues_indexes.results_tail;
        } else { // flag == HEAD
            local_dst = &queues_indexes.tasks_head;
            remote_src = remote_info.tasks_head_addr;       // remote_src
            rkey = remote_info.tasks_head_rkey;             // rkey
            wr_id = queues_indexes.tasks_head;
        }

        post_rdma_read(local_dst,                   // local_dst
                       sizeof(size_t),              // len
                       mr_queues_indexes->lkey,     // lkey
                       remote_src,                  // remote_src
                       rkey,                        // rkey
                       wr_id);                      // wr_id

        //check for CQE
        struct ibv_wc wc;
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
        } while(ncqes == 0);

        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);
        if (wc.opcode != IBV_WC_RDMA_READ) {
            exit(1);
        }
    }

    void update_index(bool flag) {
        // see about making this an atomic operation
        void *local_src = NULL;
        uint64_t remote_dst = 0;  
        uint32_t rkey = 0;    
        uint64_t wr_id = 0;
        int ncqes = 0;
        if (flag == TAIL) {
            queues_indexes.tasks_tail++;                // update local index
            local_src = &queues_indexes.tasks_tail;
            remote_dst = remote_info.tasks_tail_addr;   // remote_src
            rkey = remote_info.tasks_tail_rkey;         // rkey
            wr_id = queues_indexes.tasks_tail;
        } else { // flag == HEAD
            queues_indexes.results_head++;
            local_src = &queues_indexes.results_head;
            remote_dst = remote_info.results_head_addr; // remote_src
            rkey = remote_info.results_head_rkey;       // rkey
            wr_id = queues_indexes.results_head + remote_info.queue_size; 
        }

        post_rdma_write(
            remote_dst,                 // remote_dst
            sizeof(size_t),             // len
            rkey,                       // rkey
            local_src,                  // local_src
            mr_queues_indexes->lkey,    // lkey
            wr_id,                      // wr_id
            nullptr);  
        //check for CQE
        struct ibv_wc wc;
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
        } while(ncqes == 0);

        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }

        VERBS_WC_CHECK(wc);
        if (wc.opcode != IBV_WC_RDMA_WRITE) {
            perror("write index failed");
            exit(1);
        }
    }

    void copy_image(size_t index, uchar* src, uchar* dst, bool flag) {
        int ncqes = 0;
        if (flag == COPY_TO_SERVER) {
            // printf("copying image to server\n");
            post_rdma_write(
                (uint64_t)dst,                  // remote_dst
                IMG_SZ,                         // len
                remote_info.images_in_rkey,     // rkey
                src,                            // local_src
                mr_images_in->lkey,             // lkey
                index,                          // wr_id
                nullptr); 
        } else { // flag == COPY_FROM_SERVER
            // printf("copying image from server\n");
            post_rdma_read(
                src,                            // local_dst
                IMG_SZ,                         // len
                mr_images_out->lkey,            // lkey
                (uintptr_t)dst,                 // remote_src
                remote_info.images_out_rkey,    // rkey
                index);                         // wr_id
        }
        struct ibv_wc wc;
            
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
        } while(ncqes == 0);

        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);
        if (flag == COPY_TO_SERVER) {
            if(wc.opcode != IBV_WC_RDMA_WRITE) {
                perror("write image failed");
                exit(1);
            }
            return;
        }
        if (flag == COPY_FROM_SERVER) {
            if(wc.opcode != IBV_WC_RDMA_READ) {
                perror("read image failed");
                exit(1);
            }
            return;
        }
        perror("unexpected error");
        exit(1);
    }

    void enqueue_data_element(size_t index, data_element *data) {
        data_element *remote_tasks_queue = (data_element *)remote_info.tasks_queue_addr;
        uint64_t remote_task_addr = (uintptr_t)&remote_tasks_queue[index % remote_info.queue_size];
        //printf("index : %d\n", index);
        post_rdma_write(
            remote_task_addr,               // remote_dst
            sizeof(data_element),           // len
            remote_info.tasks_queue_rkey,   // rkey
            data,                           // local_src
            mr_sending_data_element->lkey,  // lkey
            index,                          // wr_id
            nullptr); 
            
        struct ibv_wc wc; 
        int ncqes = 0;
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
        } while(ncqes == 0);

        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);
        if (wc.opcode != IBV_WC_RDMA_WRITE) {
            perror("enqueue job failed");
            exit(1);
        }

    }

    void dequeue_data_element(size_t index, data_element *data) {
        data_element *remote_results_queue = (data_element *)remote_info.results_queue_addr;
        uint64_t remote_result_addr = (uintptr_t)&remote_results_queue[index % remote_info.queue_size];

        post_rdma_read(
            data,                           // local_dst
            sizeof(data_element),           // len
            mr_recieved_data_element->lkey, // lkey
            remote_result_addr,             // remote_src
            remote_info.results_queue_rkey, // rkey
            index);                         // wr_id
            
        struct ibv_wc wc; 
        int ncqes = 0;
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
        } while(ncqes == 0);

        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);
        if (wc.opcode != IBV_WC_RDMA_READ) {
            perror("dequeue job failed");
            exit(1);
        }
    }

    void sendTermination() {
        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send killing request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = KILL_SERVER;
        req->input_rkey = 0;
        req->input_addr = (uintptr_t)nullptr;
        req->input_length = IMG_SZ;
        req->output_rkey = 0;
        req->output_addr = (uintptr_t)nullptr;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = (uint64_t)KILL_SERVER; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }
    }

    bool getTermination(int *img_id) {
        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	    VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }
        if (*img_id != KILL_SERVER)
            printf("Unexpected request\n");
        return true;
    }

    void kill() {
        sendTermination();

        int img_id = 0;
        bool dequeued;
        do {
            dequeued = getTermination(&img_id);
        } while (!dequeued || img_id != -1);
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
