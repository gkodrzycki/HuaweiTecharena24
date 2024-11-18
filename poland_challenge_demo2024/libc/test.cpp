#include <bits/stdc++.h>

#include <filesystem>
#include <iostream>

#include "ann/third_party/helpa/helpa/core.hpp"

using namespace std;

#include "../ann/memory.hpp"
#include "ann.h"
#include "ann/neighbor.hpp"

/*
 * Measures the current (and peak) resident and virtual memories
 * usage of your linux C process, in kB
 */
int getMemory_ann() {
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
  // std::cout << "Curr. Real Mem.: " << currRealMem << ", Peak Real Mem.: " <<
  // peakRealMem << ", Curr Virt Mem.:" << currVirtMem << ", Peak Virt Mem.: "
  // << peakVirtMem << "\n";
  std::cout << "Peak Virt Mem.: " << (float)peakVirtMem / 1024 / 1024
            << "GB \n";
  return peakVirtMem;  // KB
}

int intersect(int* a1, int* a2, int l1, int l2, int index) {
  int res = 0;
  for (int i = 0; i < l1; i++) {
    if (a1[i] < 0) {
      continue;
    }

    for (int j = 0; j < l2; j++) {
      if ((j > 0) && (a2[j] == a2[j - 1])) continue;
      if (a1[i] == a2[j]) {
        res++;
        break;
      }
    }
  }
  return res;
}

const std::string metric = "IP";
const int dim = 200;
const int num_elements = 50'000;
const int num_closest = 10;
const int num_queries = 5'000;
const int seed = 47;

struct QueueData {
  float dist;
  int id;

  bool operator<(const QueueData& other) const {
    if (dist != other.dist) {
      return dist < other.dist;
    }
    return id < other.id;
  }
};

struct Dataset {
  std::vector<float> data;
  std::vector<float> queries_data;
  std::vector<int> closest;

  static Dataset Create() {
    Dataset dataset;
    dataset.data.resize(num_elements * dim);
    dataset.queries_data.resize(num_queries * dim);
    dataset.closest.resize(num_queries * num_closest);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real(-10., 10.);

    for (auto& i : dataset.data) {
      i = distrib_real(rng);
    }
    for (auto& i : dataset.queries_data) {
      i = distrib_real(rng);
    }

    // Generate closest

    auto dist_func =
        (metric == "L2" ? helpa::l2_fp32_fp32 : helpa::dot_fp32_fp32);
    float* data = dataset.data.data();
    float* queries_data = dataset.queries_data.data();

#pragma omp parallel for num_threads(6)
    for (int i = 0; i < num_queries; ++i) {
      std::priority_queue<QueueData> Q;

      for (int j = 0; j < num_elements; ++j) {
        const float dist =
            dist_func(queries_data + i * dim, data + j * dim, dim);
        Q.push({dist, j});
        if (Q.size() > num_closest) {
          // Remove the node with the highest dist
          Q.pop();
        }
      }

      // Reverse the order as the highest dist is on top.
      for (int j = num_closest - 1; j >= 0; --j) {
        dataset.closest[i * num_closest + j] = Q.top().id;
        Q.pop();
      }
    }

    return dataset;
  }

  static Dataset Read(std::string filename) {
    std::cerr << "Reading the dataset from: " + filename << "\n";
    Dataset dataset;
    dataset.data.resize(num_elements * dim);
    dataset.queries_data.resize(num_queries * dim);
    dataset.closest.resize(num_queries * num_closest);

    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.good()) {
      std::cerr << "Something not good with file: " << filename << "\n";
      exit(1);
    }

    file.read(reinterpret_cast<char*>(dataset.data.data()),
              sizeof(float) * dataset.data.size());
    file.read(reinterpret_cast<char*>(dataset.queries_data.data()),
              sizeof(float) * dataset.queries_data.size());
    file.read(reinterpret_cast<char*>(dataset.closest.data()),
              sizeof(int) * dataset.closest.size());
    file.close();
    return dataset;
  }

  void Save(std::string filename) {
    std::cerr << "Saving the dataset to file: " + filename << "\n";
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.good()) {
      std::cerr << "Something not good with file: " << filename << "\n";
      exit(1);
    }

    file.write(reinterpret_cast<const char*>(data.data()),
               sizeof(float) * data.size());
    file.write(reinterpret_cast<const char*>(queries_data.data()),
               sizeof(float) * queries_data.size());
    file.write(reinterpret_cast<const char*>(closest.data()),
               sizeof(int) * closest.size());
    file.close();
  }
};

Dataset GetDataset() {
  std::string filename = metric + ".bin";

  if (std::filesystem::exists(filename)) {
    return Dataset::Read(filename);
  }

  Dataset dataset = Dataset::Create();
  dataset.Save(filename);
  return dataset;
}

int main() {
  auto dataset = GetDataset();

  void* vidx = ann_init(dim, 50, metric.c_str());

  // building process
  double build_time = -0.0;
  {
    printf("building process test: \n");
    auto startTime = chrono::steady_clock::now();

    // Add data to index
    ann_add(vidx, num_elements, dataset.data.data(), nullptr);

    auto endTime = chrono::steady_clock::now();
    auto timeTaken =
        chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count();
    build_time = (double)timeTaken / 1000 / 1000 / 1000;
    std::cout << "Index is built successfully. Build time: " << build_time
              << "s\n";
  }

  printf("searching process test: \n");
  float* distances = new float[num_queries * num_closest]();
  int32_t* labels = new int32_t[num_queries * num_closest]();

  for (int ef : {100, 150, 200}) {
    double recall = 0;
    double qps = 0;

    set_ann_ef(vidx, ef);
    double best_time = 1e9;
    // TODO: maybe change max_iter to 3
    const int max_iter = 1;
    for (int iter = 0; iter < max_iter; ++iter) {
      auto startTime = chrono::steady_clock::now();
      ann_search(vidx, num_queries, dataset.queries_data.data(), num_closest,
                 distances, labels, /*num_p=*/8);
      auto endTime = chrono::steady_clock::now();
      auto timeTaken =
          chrono::duration_cast<chrono::nanoseconds>(endTime - startTime)
              .count();
      double timeSeconds = (double)timeTaken / 1000 / 1000 / 1000;
      best_time = std::min(best_time, timeSeconds);
    }

    int found = 0;
    for (int i = 0; i < num_queries; ++i) {
      found += intersect(dataset.closest.data() + i * num_closest,
                         labels + i * num_closest, num_closest, num_closest, i);
    }

    qps = (double)num_queries / best_time;
    recall = (double)found / (num_queries * num_closest);

    std::cout << "*************************************\n";
    std::cout << "EF: " << ef << "\n";
    int peakmem = getMemory_ann();
    cout << "Build time: " << build_time << ", Peak memory : " << peakmem
         << "\n";
    cout << "Recall: " << recall << ", QPS: " << qps << "\n";
  }
}