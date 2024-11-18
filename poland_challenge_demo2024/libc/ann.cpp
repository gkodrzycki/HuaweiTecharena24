#include "ann.h"
#include <thread>
#include <omp.h>
#include <iostream>

#include "ann/builder.hpp"
#include "ann/nsg/nsg.hpp"
#include "ann/hnsw/hnsw.hpp"
#include "ann/searcher/graph_searcher.hpp"

using IndexNSG = ann::NSG;
std::unique_ptr<ann::GraphSearcherBase> searcher;
std::unique_ptr<ann::Builder> graph_builder;

int dimension = 0;
std::string metrica = "";

std::string quant_build = "";
std::string quant_search = "";



bool is_load = 0;

void* ann_init(int K_features, int R, const char* metric) {
    dimension = K_features;
    metrica = metric;

    int annR, annL;

    if(metrica == "L2") {
        annR = 100;
        annL = 200;
        quant_build = "SQ4U";
        quant_search = "SQ4U";
        graph_builder = ann::create_hnsw(metrica, quant_build, dimension, annR, annL);
    } 
    else {
        annR = 100;
        annL = 200;
        quant_build = "SQ4U";
        quant_search = "SQ8U";

        graph_builder = std::unique_ptr<IndexNSG>(
            new IndexNSG(dimension, metrica, annR, annL)
        );

        // graph_builder = ann::create_hnsw(metrica, quant_build, dimension, annR, annL);
    } 


    return graph_builder.get();
}

void ann_free(void* ptr) {
    graph_builder.reset();
    searcher.reset();
}

void ann_add(void* ptr, int n, float* x, const char* store) {

    std::cerr << "Building the index\n";
    if(!is_load) {
        graph_builder->Build(x, n);
    }

    std::cerr << "Graph is built\n";

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
    searcher->SetEf(ann_ef);
}

void ann_search(void* ptr, int n, const float* x, int k, float* distances, int32_t* labels, int num_p) {
    if (!searcher) return;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        const float* query = x + i * dimension; 
        float* dist_ptr = distances + i * k;
        int32_t* label_ptr = labels + i * k;
        searcher->Search(query, k, label_ptr, dist_ptr);
    }
}

void ann_load(void* ptr, const char* path) {
    graph_builder->GetGraph().load(std::string(path));
    is_load = 1;

}

void ann_save(void* ptr, const char* path) {
    graph_builder->GetGraph().save(std::string(path));
}
