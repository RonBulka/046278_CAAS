#!/bin/bash

# Run the program with streams and different load factors
run_streams () {
    local mode="streams"
    echo "Running ex2 with $mode" > ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 214.4 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 278.72 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 343.04 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 407.36 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 471.68 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 536 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 600.32 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 664.64 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 728.96 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 793.28 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode 857.6 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
}

# Run the program with queue and different load factors and number of threads
run_queue () {
    local mode="queue"
    echo "Running ex2 with $mode and $1 threads" > ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 214.4 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 278.72 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 343.04 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 407.36 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 471.68 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 536 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 600.32 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 664.64 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 728.96 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 793.28 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
    ./ex2 $mode $1 857.6 >> ex2_$mode.txt
    echo "-----------------------------------------------------------------------" >> ex2_$mode.txt
}

# Compile the program
make ex2

# Run the program with streams
run_streams

# Run the program with queue and different number of threads
# ./ex2 queue 1024 0 > output_queue.txt
# run_queue 1024
# run_queue 512
# run_queue 256