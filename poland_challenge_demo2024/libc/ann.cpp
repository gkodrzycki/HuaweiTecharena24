#include "ann.h"
#include <thread>
#include <omp.h>

#include "ann/builder.hpp"
#include "ann/hnsw/hnsw.hpp"
#include "ann/searcher/graph_searcher.hpp"

// using IndexHNSW = ann::HNSW<ann::SQ6Quantizer<ann::Metric::L2>>;
std::unique_ptr<ann::GraphSearcherBase> searcher;
std::unique_ptr<ann::Builder> graph_builder;

int dimension = 0;
std::string metrica = "";


void* ann_init(int K_features, int R, const char* metric) {
    dimension = K_features;
    metrica = metric;
    // R = 96;
    graph_builder = ann::create_hnsw(metric, "SQ8U", dimension, R, R+100);

    return graph_builder.get();
}

void ann_free(void* ptr) {
    // if (ptr) {
    //     auto* index = static_cast<IndexHNSW*>(ptr);
    //     delete index;using IndexHNSW = ann::HNSW<ann::SQ6Quantizer<ann::Metric::L2>>;

    // }
    graph_builder.reset();
    searcher.reset();
}

void ann_add(void* ptr, int n, float* x, const char* store) {
    graph_builder->Build(x, n);

    auto graph = graph_builder->GetGraph();

    if (store != nullptr) {
        graph.save(std::string(store));
    }

    searcher = std::move(ann::create_searcher(
        std::move(graph),
        metrica,
        "SQ6"  
    ));

    searcher->SetData(x, n, dimension);
}

void set_ann_ef(void* ptr, int ann_ef) {
    searcher->SetEf(ann_ef);
}

bool optimize_once = true;
void ann_search(void* ptr, int n, const float* x, int k, float* distances, int32_t* labels, int num_p) {
    if (!searcher) return;

    num_p *= 20;
    // if(optimize_once){
    //     searcher->Optimize(num_p);
    //     optimize_once = false;
    // }

    #pragma omp parallel for num_threads(num_p)
    for (int i = 0; i < n; ++i) {
        const float* query = x + i * dimension; 
        float* dist_ptr = distances + i * k;
        int32_t* label_ptr = labels + i * k;
        searcher->Search(query, k, label_ptr, dist_ptr);
    }
}

void ann_load(void* ptr, const char* path) {
    graph_builder->GetGraph().load(std::string(path));
}

void ann_save(void* ptr, const char* path) {
    graph_builder->GetGraph().save(std::string(path));
}
