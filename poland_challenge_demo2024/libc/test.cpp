/*20240108 w00678802*/
#include <bits/stdc++.h>

using namespace std;

// #if defined(__aarch64__)
// #include "hdf5.h"
// #else
// #include <hdf5.h>
// #endif
#include "../ann/memory.hpp"
#include "ann.h"

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

int main() {
  // Generate random dataset.

  int dim = 200;
  int max_elements = 100'000;

  // Generate random data
  std::mt19937 rng;
  rng.seed(47);
  std::uniform_real_distribution<> distrib_real;

  float* data = new float[dim * max_elements];
  for (int i = 0; i < dim * max_elements; i++) {
    data[i] = distrib_real(rng);
  }

  std::string metric = "L2";
  void* vidx = ann_init(dim, 50, metric.c_str());

  // building process
  double build_time = -0.0;
  {
    printf("building process test: \n");
    auto startTime = chrono::steady_clock::now();

    // Add data to index
    cout << "HERE\n";
    ann_add(vidx, max_elements, data, nullptr);
    cout << "HERE2\n";
    auto endTime = chrono::steady_clock::now();
    auto timeTaken =
        chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count();
    build_time = (double)timeTaken / 1000 / 1000 / 1000;
    std::cout << "Index is built successfully. Build time: " << build_time
              << "s\n";
  }

  int num_closest = 1;

  printf("searching process test: \n");
  float* distances = new float[num_closest]();
  int32_t* labels = new int32_t[num_closest]();
  double recall = 0;
  double qps = 0;

  auto startTime = chrono::steady_clock::now();
  int correct = 0;
  for (int i = 0; i < max_elements; ++i) {
    // Search for this one vector
    ann_search(vidx, 1, data + i * dim, num_closest, distances, labels, 1);

    if (labels[0] == i) {
      correct += 1;
    }
  }
  auto endTime = chrono::steady_clock::now();
  auto timeTaken =
      chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count();
  double timeSeconds = (double)timeTaken / 1000 / 1000 / 1000;

  qps = (double)max_elements / timeSeconds;
  recall = (double)correct / max_elements;

  int peakmem = getMemory_ann();
  cout << "Build time: " << build_time << ", Peak memory : " << peakmem << "\n";
  cout << "Recall: " << recall << ", QPS: " << qps << "\n";
}