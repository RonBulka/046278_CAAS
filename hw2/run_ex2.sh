#!/bin/bash
# Run the program with streams and 0 load factor
run_streams_0 () {
    local mode="streams"
    echo "Running ex2 with ${mode} and load 0" > ex2_${mode}_0.txt
    for i in {1..10}
    do
        echo "-----------------------------------------------------------------------" >> ex2_${mode}_0.txt
        ./ex2 $mode 0 >> ex2_${mode}_0.txt
    done
}

# Run the program with streams and different load factors
run_streams () {
    local mode="streams"
    echo "Running ex2 with ${mode}" > ex2_${mode}.txt
    for i in {0..10}
    do
        local load=$(echo "scale=2; $i * 4894.172 + 2575.88" | bc)
        echo "-----------------------------------------------------------------------" >> ex2_${mode}.txt
        ./ex2 $mode $load >> ex2_${mode}.txt
    done
}

# Run the program with queue and different load factors and number of threads
run_queue_0 () {
    local mode="queue"
    echo "Running ex2 with ${mode} and $1 threads and load 0" > ex2_${mode}_0.txt
    for i in {0..10}
    do
        echo "-----------------------------------------------------------------------" >> ex2_${mode}_0.txt
        ./ex2 $mode $1 0 >> ex2_${mode}_0.txt
    done
}

# Run the program with queue and different load factors and 1024 threads
run_queue_1024 () {
    local mode="queue"
    echo "Running ex2 with ${mode} and 1024 threads" > ex2_${mode}_1024.txt
    for i in {0..10}
    do
        local load=$(echo "scale=2; $i * 5562.041 + 2927.39" | bc)
        echo "-----------------------------------------------------------------------" >> ex2_${mode}_1024.txt
        ./ex2 $mode 1024 $load >> ex2_${mode}_1024.txt
    done
}

# Run the program with queue and different load factors and 512 threads
run_queue_512 () {
    local mode="queue"
    echo "Running ex2 with ${mode} and 512 threads" > ex2_${mode}_512.txt
    for i in {0..10}
    do
        local load=$(echo "scale=2; $i * 15148.7 + 7973" | bc)
        echo "-----------------------------------------------------------------------" >> ex2_${mode}_512.txt
        ./ex2 $mode 512 $load >> ex2_${mode}_512.txt
    done
}

# Run the program with queue and different load factors and 256 threads
run_queue_256 () {
    local mode="queue"
    echo "Running ex2 with ${mode} and 256 threads" > ex2_${mode}_256.txt
    for i in {0..10}
    do
        local load=$(echo "scale=2; $i * 19801.781 + 10421.99" | bc)
        echo "-----------------------------------------------------------------------" >> ex2_${mode}_256.txt
        ./ex2 $mode 256 $load >> ex2_${mode}_256.txt
    done
}

clean_up () {
    # make clean
    rm -f ex2_streams_0.txt
    rm -f ex2_streams.txt
    rm -f ex2_queue.txt
}

# Compile the program
make ex2

# Run the program with streams
# run_streams_0
# run_streams

# Run the program with queue and different number of threads
# run_queue_0 1024
# run_queue_1024
# run_queue_0 512
# run_queue_512
# run_queue_0 256
run_queue_256

# Clean up
# clean_up