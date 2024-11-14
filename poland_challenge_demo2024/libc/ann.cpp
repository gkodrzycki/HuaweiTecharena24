#include "ann.h"
#include <thread>
#include <omp.h>
#include <iostream>

#include "ann/builder.hpp"
#include "ann/hnsw/hnsw.hpp"
#include "ann/searcher/graph_searcher.hpp"

std::unique_ptr<ann::GraphSearcherBase> searcher;
std::unique_ptr<ann::Builder> graph_builder;

int dimension = 0;
std::string metrica = "";

std::string quant_build = "";
std::string quant_search = "";

#define num_of_threads 96

void* ann_init(int K_features, int R, const char* metric) {
    dimension = K_features;
    metrica = metric;

    int annR, annL;

    if(metrica == "L2") {
        annR = 50;
        annL = 200;
        quant_build = "SQ4U";
        quant_search = "SQ4U";
    } 
    else {
        annR = 50;
        annL = 200;
        quant_build = "SQ8P";
        quant_search = "SQ8P";
    } 

    graph_builder = ann::create_hnsw(metric, quant_build, dimension, annR, annL);

    return graph_builder.get();
}

void ann_free(void* ptr) {
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
        quant_search  
    ));

    searcher->SetData(x, n, dimension);

    searcher->Optimize();
}

void set_ann_ef(void* ptr, int ann_ef) {
    if(metrica == "IP") 
        ann_ef /= 2;
    searcher->SetEf(ann_ef);
}

bool optimize_once = true;
void ann_search(void* ptr, int n, const float* x, int k, float* distances, int32_t* labels, int num_p) {
    if (!searcher) return;

    #pragma omp parallel for num_threads(num_of_threads)
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
