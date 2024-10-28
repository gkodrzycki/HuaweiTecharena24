
All the contestant's source files should be in the ann folder. And any modification of main.cpp is forbidden.


# PASS condition
- recall >= 0.95
- peakmem <= mem_bound
- build_time <= time_bound
The value time_bound needs to be modified for a given machine, where time_bound >=1.2*build_time is expected.

# Evaluation
rank method:
    1. qps more is better.
    2. build time less is better.
    3. recall more is better.
    4. peak memory less is better.   
priority: 1>2>3>4
        

# Compiling and Running Methods
1. cd poland_challenge_demo2024/libc && mkdir build && cd build && cmake .. && make -j
2. ./run



# Datasets
The code of loading dataset 1 is below the comment /*dataset 1*/. The code of loading dataset 2 is below the comment /*dataset 2*/. Please verify the data_path and choose the related datasets before compiling the programm. 