/*20240108 w00678802*/
#include <bits/stdc++.h>

using namespace std;

// #if defined(__aarch64__)
// #include "hdf5.h"
// #else
// #include <hdf5.h>
// #endif
#include "ann.h"

#include"../ann/memory.hpp"


static const char *HDF5_DATASET_TRAIN = "train";
static const char *HDF5_DATASET_TEST = "test";
static const char *HDF5_DATASET_NEIGHBORS = "neighbors";
static const char *HDF5_DATASET_DISTANCES = "distances";

// void *hdf5_read(const std::string &file_name, const std::string &dataset_name, H5T_class_t dataset_class,
//                 int32_t &d_out, int32_t &n_out);
template<typename T, typename U> void read_dataset(std::string dataset_filename, uint32_t &numVectors, uint32_t &numDimensions, U* &data);
int getMemory_ann();
double distance(const float *x, const float *y, int d, const std::string metric)
{
    if (metric == "L2") {
        double sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += (double)(x[i] - y[i]) * (x[i] - y[i]);
        }
        return sum;
    } else if (metric == "IP") {
        double sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum -= (double)x[i] * y[i];
        }
        return sum;
    } else {
        assert(false);
    }
    return 0;
}

int intersect(int64_t *a1, int32_t *a2, int l1, int l2, int index)
{
    int res = 0;
    for (int i = 0; i < l1; i++) {
        if (a1[i] < 0) {
            continue;
        }

        for (int j = 0; j < l2; j++) {
            if( (j > 0) && (a2[j] == a2[j-1])) continue;
            if (a1[i] == a2[j]) {
                res++;
                break;
            }
        }
    }
    return res;
}

void **numpy_read(const std::string &fileName, int32_t &dim, int32_t &number, int consecutiveLog, int &numAllocParts)
{
    std::ifstream reader(fileName.c_str(), std::ios::binary);
    reader.ignore(8);
    uint16_t metaLen;
    reader.read((char *)&metaLen, 2);
    char metaArr[metaLen + 1] = {0};
    reader.read(metaArr, metaLen);
    std::string meta(metaArr, metaLen);

    int beg = meta.find_last_of('(');
    int end = meta.find_last_of(')');
    std::string dims = meta.substr(beg + 1, end);
    std::string dim1 = dims.substr(0, dims.find(','));
    std::string dim2 = dims.substr(dims.find(',') + 2);
    number = stoi(dim1);
    dim = stoi(dim2);

    numAllocParts = (number + (1ULL << consecutiveLog) - 1) >> consecutiveLog;
    void **data = new void*[numAllocParts];
    if (meta.find("<f4") != string::npos) { // float
        for (int i = 0 ; i < numAllocParts; i++) {
            uint64_t toRead = std::min((1ULL << consecutiveLog), number - i * (1ULL << consecutiveLog));
            data[i] = new float[toRead * dim];
            reader.read((char *)data[i], toRead * dim * sizeof(float));
        }
    } else if (meta.find("<f8") != string::npos) { // double
        for (int i = 0 ; i < numAllocParts; i++) {
            uint64_t toRead = std::min((1ULL << consecutiveLog), number - i * (1ULL << consecutiveLog));
            data[i] = new double[toRead * dim];
            reader.read((char *)data[i], toRead * dim * sizeof(double));
        }
    } else if (meta.find("<i4") != string::npos) { // int32_t
        for (int i = 0 ; i < numAllocParts; i++) {
            uint64_t toRead = std::min((1ULL << consecutiveLog), number - i * (1ULL << consecutiveLog));
            data[i] = new int32_t[toRead * dim];
            reader.read((char *)data[i], toRead * dim * sizeof(int32_t));
        }
    } else if (meta.find("<i8") != string::npos) { // int64_t
        for (int i = 0 ; i < numAllocParts; i++) {
            uint64_t toRead = std::min((1ULL << consecutiveLog), number - i * (1ULL << consecutiveLog));
            data[i] = new int64_t[toRead * dim];
            reader.read((char *)data[i], toRead * dim * sizeof(int64_t));
        }
    } else {
        printf("error, can't find type in npy array!\n");
        exit(1);
    }

     if (reader.fail()) {
        printf("error while reading npy array!\n");
        exit(1);
    }
    return data;
}

void loadNPY(const std::string &dataset_folder, int32_t &nb, int32_t &nq, int32_t &dim, int32_t &gt_closest,
    float *&data, float *&queries, int64_t *&gt_ids, int consecutiveLog, int &numAllocParts)
{
    std::string base = dataset_folder + "/base.npy";
    std::string query = dataset_folder + "/query.npy";
    std::string groud_truth = dataset_folder + "/gt.npy";
    int tmp;
    float ** data2 = (float **)numpy_read(base, dim, nb, consecutiveLog, numAllocParts);
    float **queries2 = (float **)numpy_read(query, dim, nq, consecutiveLog, tmp);
    int64_t **gt_ids2 = (int64_t **)numpy_read(groud_truth, gt_closest, nq, consecutiveLog, tmp);
    queries = queries2[0];
    gt_ids = gt_ids2[0];
    data = data2[0];

    delete[] queries2;
    delete[] gt_ids2;
    delete[] data2;

}

// void loadHDF(const std::string &ann_file_name, int32_t &nb, int32_t &nq, int32_t &dim, int32_t &gt_closest,
//     float *&data, float *&queries, int64_t *&gt_ids, int consecutiveLog, int &numAllocParts)
// {
//     data = (float *)hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb);
//     queries = (float *)hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq);
//     int32_t *gt_ids_short = (int32_t *)hdf5_read(ann_file_name, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, gt_closest, nq);
//     gt_ids = new int64_t[gt_closest * nq];
//     for (int i = 0; i < gt_closest * nq; i++) {
//         gt_ids[i] = gt_ids_short[i];
//     }

//     int cnt = (nb + (1ULL << consecutiveLog) - 1) >> consecutiveLog;
//     // data = new float*[cnt];
//     // for (int i = 0; i < cnt; i++) {
//     //     data[i] = &data_one[dim * i * (1ULL << consecutiveLog)];
//     // }
//     numAllocParts = 1;
//     delete[] gt_ids_short;
// }

template <typename data_t>
void* readBin(const std::string &filename, int32_t &dim, int32_t &number){

    std::ifstream reader(filename.c_str(), std::ios::binary);
    if (!reader.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    reader.read((char *)&number, sizeof(uint32_t));
    reader.read((char *)&dim,sizeof(uint32_t));

    data_t * data1 = (data_t *) ann::align_alloc((int64_t)number * dim * sizeof(data_t));

    reader.read((char *)data1, (int64_t)number * dim * sizeof(data_t));
    reader.close();
    std::cout << "n * d = " << number <<" * " << dim <<std::endl;
    data_t max = -9999;
    data_t min = 9999;
    for(size_t i = 0; i < dim * number; i++){
        data_t x = data1[i];
        // max = max>x?max:x;
        // min = min<x?min:x;
        if(x> max) max = x;
        if(x< min) min = x;
    }
    std::cout << "max: " << max <<", min: " << min <<std::endl;

    return data1;

}


double time_bound = 1000; //to do: decide by final machine *1.5, stop if the time is more than time_bound + 3seconds.
int mem_bound = 1024*1024*25; //KB,to do: decide by test *1.5

const int consecutiveLog = 20;
int numAllocParts = 0;

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset number>\n";
        exit(1);
    }

    float *xb_;
    float *xq_;
    int64_t *gt_ids_;
    uint32_t nb_, nq_, dim_, gt_closest;
    int closestNum = 10;
    int num_p = 8;
    std::string metric ="L2";

    printf("loading data : \n");
    getMemory_ann();

    /*dataset 1*/
    if (argv[1][0] == '1') {
        std::string dataset = "/TechArena_Poland2024_datasets/dataset2M200/";
        std::string xb_file = dataset + "base_2M.bin";
        std::string xq_file = dataset + "query_10K.bin";
        std::string gt_file = dataset + "gt_10K.bin";
        read_dataset<uint8_t, float>( xb_file, nb_, dim_, xb_);
        read_dataset<uint8_t, float>(xq_file, nq_, dim_, xq_);
        read_dataset<uint32_t, int64_t>(gt_file, nq_, gt_closest, gt_ids_);
    }
    /*dataset 2*/
    else if (argv[1][0] == '2') {
		std::string dataset = "/TechArena_Poland2024_datasets/dataset2M384/";
        metric ="IP";

        std::string xb_file = dataset + "base_2M.bin";
        std::string xq_file = dataset + "query_10K.bin";
        std::string gt_file = dataset + "gt_10K.bin";
        read_dataset<float, float>( xb_file, nb_, dim_, xb_);
        read_dataset<float, float>(xq_file, nq_, dim_, xq_);
        read_dataset<uint32_t, int64_t>(gt_file, nq_, gt_closest, gt_ids_);
    } else {
        std::cerr << "Usage: " << argv[0] << " <dataset number>\n";
        exit(1);
    }

    getMemory_ann();

    int R = 50;
    void *vidx = ann_init(dim_, R, metric.c_str());
    uint64_t timeTaken = 0;
    chrono::_V2::steady_clock::time_point startTime, endTime;

    //building process
    double build_time = -0.0;
    {
        printf("building process test: \n");
        startTime = chrono::steady_clock::now();

        ann_add(vidx, nb_, xb_,  NULL);

        endTime = chrono::steady_clock::now();
        timeTaken = chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count();
        build_time = (double)timeTaken / 1000 / 1000 / 1000;
        std::cout << "Index is built successfully. Build time: " << build_time << "s\n";


    }




    getMemory_ann();

    //searching process
    const int minEF = 100;
    const int maxEF = 400;
    const int efStep = 20;
    int numIters =3;
    {
        printf("searching process test: \n");
        float *distances = new float[nq_ * closestNum]();
        int32_t *labels = new int32_t[nq_ * closestNum]();
        double recall =0;
        double qps =0;

        for (int ef = minEF; ef <= maxEF; ef += efStep) {

            set_ann_ef(vidx, ef);
            double totalTime = 0;
            double bestTime = 1e9;

            for (int iter = 0; iter < numIters; iter++) {

                startTime = chrono::steady_clock::now();

                ann_search(vidx, nq_, xq_, closestNum, distances, labels,num_p);

                endTime = chrono::steady_clock::now();
                timeTaken = chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count();
                double timeSeconds = (double)timeTaken / 1000 / 1000 / 1000;

                totalTime += timeSeconds;
                bestTime = min(bestTime, timeSeconds);
                //std::cout << "ann searching finished, time: " << timeSeconds<< "s, queries per second: " << (double)nq_ / timeSeconds << "\n";
            }

            //compute recall
            int found = 0;
            int gtWanted = 10;
            for (int i = 0; i < nq_; i++) {
                found += intersect(gt_ids_ + gt_closest * i, labels + closestNum * i, gtWanted, closestNum, i);
            }
            qps = (double)nq_ / bestTime;
            recall = (double)found / nq_ / gtWanted;
            cout << "ef = "<< ef <<", recall = "<< recall <<", score = "<< qps << "\n";
            if(recall > 0.95) break;


        }


        //score
        int peakmem = getMemory_ann();
        cout<< "Build time: " <<build_time <<", Peak memory : "<< peakmem  << "\n";
        cout<< "Recall: " <<recall<<", QPS: "<< qps << "\n";
        if( recall < 0.95 ){
            cout <<"Recall_10@10 = " << recall << "0.95, score = 0 \n";
        }
        else if(peakmem > mem_bound){
            cout << "The peak of memory used: "<< peakmem<< "  is out of bound"<< mem_bound <<" , score = 0 \n";
        }else if (build_time > time_bound ) {
            std::cout << "Time out of bound: " <<  time_bound << "s\n";
        }else{
            cout <<"recall = "<< recall <<", score = "<< qps << "\n";
        }

        /* rank method:
        1. qps more is better.
        2. build time less is better.
        3. recall more is better.
        4. peak memory less is better.

        priority: 1>2>3>4
        */


        delete[] distances;
        delete[] labels;
    }


    printf("The test is completed. \n");


    delete[] xq_;
    delete[] xb_;
    delete[] gt_ids_;



    return 0;
}

// void *hdf5_read(const std::string &file_name, const std::string &dataset_name, H5T_class_t dataset_class,
//                 int32_t &d_out, int32_t &n_out)
// {
//     hid_t file, dataset, datatype, dataspace, memspace;
//     H5T_class_t t_class;      /* data type class */
//     hsize_t dimsm[3];         /* memory space dimensions */
//     hsize_t dims_out[2];      /* dataset dimensions */
//     hsize_t count[2];         /* size of the hyperslab in the file */
//     hsize_t offset[2];        /* hyperslab offset in the file */
//     hsize_t count_out[3];     /* size of the hyperslab in memory */
//     hsize_t offset_out[3];    /* hyperslab offset in memory */
//     void *data_out = nullptr; /* output buffer */

//     /* Open the file and the dataset. */
//     file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
//     dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);

//     /* Get datatype and dataspace handles and then query
//      * dataset class, order, size, rank and dimensions. */
//     datatype = H5Dget_type(dataset); /* datatype handle */
//     t_class = H5Tget_class(datatype);
//     // assert(t_class == dataset_class || !"Illegal dataset class type");

//     dataspace = H5Dget_space(dataset); /* dataspace handle */
//     H5Sget_simple_extent_dims(dataspace, dims_out, nullptr);
//     n_out = dims_out[0];
//     d_out = dims_out[1];

//     /* Define hyperslab in the dataset. */
//     offset[0] = offset[1] = 0;
//     count[0] = dims_out[0];
//     count[1] = dims_out[1];
//     H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

//     /* Define the memory dataspace. */
//     dimsm[0] = dims_out[0];
//     dimsm[1] = dims_out[1];
//     dimsm[2] = 1;
//     memspace = H5Screate_simple(3, dimsm, nullptr);

//     /* Define memory hyperslab. */
//     offset_out[0] = offset_out[1] = offset_out[2] = 0;
//     count_out[0] = dims_out[0];
//     count_out[1] = dims_out[1];
//     count_out[2] = 1;
//     H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, nullptr, count_out, nullptr);

//     /* Read data from hyperslab in the file into the hyperslab in memory and display. */
//     switch (t_class) {
//         case H5T_INTEGER:
//             data_out = new int32_t[dims_out[0] * dims_out[1]];
//             H5Dread(dataset, H5T_NATIVE_INT32, memspace, dataspace, H5P_DEFAULT, data_out);  // read error

//             break;
//         case H5T_FLOAT:
//             data_out = new float[dims_out[0] * dims_out[1]];
//             H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data_out);
//             break;
//         default:
//             printf("Illegal dataset class type\n");
//             break;
//     }

//     /* Close/release resources. */
//     H5Tclose(datatype);
//     H5Dclose(dataset);
//     H5Sclose(dataspace);
//     H5Sclose(memspace);
//     H5Fclose(file);

//     return data_out;
// }





/*
 * Measures the current (and peak) resident and virtual memories
 * usage of your linux C process, in kB
 */
int getMemory_ann()
{
    int currRealMem;
    int peakRealMem;
    int currVirtMem;
    int peakVirtMem;
    // stores each word in status file
    char buffer[1024] = "";

    // linux file contains this-process info
    FILE* file = fopen("/proc/self/status", "r");

    // read the entire file
    while (fscanf(file, " %1023s", buffer) == 1) {
        if (strcmp(buffer, "VmRSS:") == 0) {
            fscanf(file, " %d", &currRealMem);
        }
        if (strcmp(buffer, "VmHWM:") == 0) {
            fscanf(file, " %d", &peakRealMem);
        }
        if (strcmp(buffer, "VmSize:") == 0) {
            fscanf(file, " %d", &currVirtMem);
        }
        if (strcmp(buffer, "VmPeak:") == 0) {
            fscanf(file, " %d", &peakVirtMem);
        }
    }
    fclose(file);
    // std::cout << "Curr. Real Mem.: " << currRealMem << ", Peak Real Mem.: " << peakRealMem << ", Curr Virt Mem.:" << currVirtMem << ", Peak Virt Mem.: " << peakVirtMem << "\n";
    std::cout << "Peak Virt Mem.: " << (float)peakVirtMem/1024/1024 << "GB \n";
    return peakVirtMem;//KB
}






template<typename T, typename U> void read_dataset(std::string dataset_filename, uint32_t &numVectors, uint32_t &numDimensions, U* &data) {
    std::ifstream reader(dataset_filename.c_str(), std::ios::binary);
    reader.read((char*)&numVectors, sizeof(uint32_t));
    reader.read((char*)&numDimensions, sizeof(uint32_t));
    T* data_T = new T[numVectors * numDimensions];
    reader.read((char*)data_T, sizeof(T) * numVectors * numDimensions);
    reader.close();

    data = new U[numVectors * numDimensions];
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < (int64_t)numVectors; ++i)
    {
        for (int64_t j = 0; j < (int64_t)numDimensions; ++j)
        {
            U valFloat = (U)data_T[i * numDimensions + j];
            std::memcpy((char *)(data + i * numDimensions + j), (char *)&valFloat, sizeof(U));
        }
    }
    delete[] data_T;
}

// int main(int argc, char** argv) {
//     if(argc < 6) {
//         std::cerr << "Usage: " << argv[0] << " <base_filename> <query_filename> <gt_filename> <base_and_query_type> <gt_type>\n";
//         exit(1);
//     }

//     std::string base_filename = argv[1];
//     std::string query_filename = argv[2];
//     std::string gt_filename = argv[3];
//     std::string base_and_query_type = argv[4];
//     std::string gt_type = argv[5];

//     float* xb_;
//     uint32_t nb_;
//     uint32_t dim_b_;

//     float* xq_;
//     uint32_t nq_;
//     uint32_t dim_q_;

//     uint32_t* x_gt_;
//     uint32_t n_gt_;
//     uint32_t dim_gt_;

//     //Add more if else depending on the other data types
//     if(base_and_query_type == "uint8_t") {
//         read_dataset<uint8_t, float>(base_filename, nb_, dim_b_, xb_);
//         read_dataset<uint8_t, float>(query_filename, nq_, dim_q_, xq_);
//     }
//     else if(base_and_query_type == "uint32_t") {
//         read_dataset<uint32_t, float>(base_filename, nb_, dim_b_, xb_);
//         read_dataset<uint32_t, float>(query_filename, nq_, dim_q_, xq_);
//     }
//     else if(base_and_query_type == "int8_t") {
//         read_dataset<int8_t, float>(base_filename, nb_, dim_b_, xb_);
//         read_dataset<int8_t, float>(query_filename, nq_, dim_q_, xq_);
//     }
//     else if(base_and_query_type == "int32_t") {
//         read_dataset<int32_t, float>(base_filename, nb_, dim_b_, xb_);
//         read_dataset<int32_t, float>(query_filename, nq_, dim_q_, xq_);
//     }
//     else {
//         //Default is float
//         read_dataset<float, float>(base_filename, nb_, dim_b_, xb_);
//         read_dataset<float, float>(query_filename, nq_, dim_q_, xq_);
//     }

//     read_dataset<uint32_t, uint32_t>(gt_filename, n_gt_, dim_gt_, x_gt_);

//     std::cout << "nb_ " << nb_ << ", nq_ " << nq_ << ", n_gt_" << n_gt_ << ", dim_b_ " << dim_b_ << ", dim_q_ " << dim_q_ << ", dim_gt_ " << dim_gt_ << "\n\n";

//     std::cout << "(base) Sample Vector Print:\n";
//     for(int i = 0; i < 384; ++i) {
//         std::cout << xb_[i] << " ";
//     }
//     std::cout << "\n\n";

//     std::cout << "(query) Sample Vector Print:\n";
//     for(int i = 0; i < 384; ++i) {
//         std::cout << xq_[i] << " ";
//     }
//     std::cout << "\n\n";

//     std::cout << "(gt) Sample Vector Print:\n";
//     for(int i = 0; i < 10; ++i) {
//         std::cout << x_gt_[i] << " ";
//     }
//     std::cout << "\n";

//     delete[] xb_;
//     delete[] xq_;
//     delete[] x_gt_;
//     return 0;
// }