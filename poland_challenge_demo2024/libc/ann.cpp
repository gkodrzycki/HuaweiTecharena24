#include "ann.h"
#include <thread>
#include <omp.h>

#include "ann/builder.hpp"
#include "ann/hnsw/hnsw.hpp"
#include "ann/searcher/graph_searcher.hpp"

using IndexHNSW = ann::HNSW<ann::FP32Quantizer<ann::Metric::L2>>;
std::unique_ptr<ann::GraphSearcherBase> searcher;

int dimension = 0;
std::string metrica = "";


void* ann_init(int K_features, int R, const char* metric) {
    // printf("%d %s\n", K_features, metric);
    dimension = K_features;
    metrica = metric;
    auto* index = new IndexHNSW(K_features, R, R + 50);
    return index;
}

void ann_free(void* ptr) {
    if (ptr) {
        auto* index = static_cast<IndexHNSW*>(ptr);
        delete index;
    }
    searcher.reset();
}

void ann_add(void* ptr, int n, float* x, const char* store) {
    auto index = static_cast<IndexHNSW*>(ptr);
    index->Build(x, n);

    auto graph = index->GetGraph();

    if (store != nullptr) {
        graph.save(std::string(store));
    }

    searcher = std::move(ann::create_searcher(
        std::move(graph),
        metrica,
        "SQ8U"  
    ));

    searcher->SetData(x, n, dimension);
}

void set_ann_ef(void* ptr, int ann_ef) {
    auto index = static_cast<IndexHNSW*>(ptr);
    index->hnsw->ef_ = ann_ef;
}

void ann_search(void* ptr, int n, const float* x, int k, float* distances, int32_t* labels, int num_p) {
    if (!searcher) return;

    #pragma omp parallel for num_threads(num_p)
    for (int i = 0; i < n; ++i) {
        const float* query = x + i * dimension; 
        float* dist_ptr = distances + i * k;
        int32_t* label_ptr = labels + i * k;
        // printf("IN\n");
        searcher->Search(query, k, label_ptr, dist_ptr);
        // printf("OUT\n");
    }
}

void ann_load(void* ptr, const char* path) {
    auto index = static_cast<IndexHNSW*>(ptr);
    index->GetGraph().load(std::string(path));
}

void ann_save(void* ptr, const char* path) {
    auto index = static_cast<IndexHNSW*>(ptr);
    index->GetGraph().save(std::string(path));
}
