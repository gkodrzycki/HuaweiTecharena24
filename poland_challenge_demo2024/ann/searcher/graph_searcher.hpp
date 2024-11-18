#pragma once

#include "ann/common.hpp"
#include "ann/graph.hpp"
#include "ann/neighbor.hpp"
#include "ann/quant/product_quant.hpp"
#include "ann/quant/quant.hpp"
#include "ann/quant/quant_base.hpp"
#include "ann/searcher/refiner.hpp"
#include "ann/searcher/searcher_base.hpp"
#include "ann/utils.hpp"
#include <algorithm>
#include <omp.h>
#include <random>

namespace ann
{

  namespace params
  {

    constexpr inline bool SQ8_REFINE = true;
    constexpr inline bool SQ8U_REFINE = true;
    constexpr inline bool SQ8P_REFINE = true;
    constexpr inline bool SQ4U_REFINE = true;
    constexpr inline bool SQ4UA_REFINE = true;
    constexpr inline bool PQ8_REFINE = true;

    constexpr inline int32_t SQ8_REFINE_FACTOR = 10;
    constexpr inline int32_t SQ8U_REFINE_FACTOR = 2;
    constexpr inline int32_t SQ8P_REFINE_FACTOR = 2;
    constexpr inline int32_t SQ4U_REFINE_FACTOR = 2;
    constexpr inline int32_t SQ4UA_REFINE_FACTOR = 10;
    constexpr inline int32_t PQ8_REFINE_FACTOR = 10;

    template <Metric metric>
    using RefineQuantizer = FP16Quantizer<metric>;

  } // namespace params

  template <QuantConcept Quant>
  struct GraphSearcher : public GraphSearcherBase
  {

    int32_t d;
    int32_t nb;
    Graph<int32_t> graph;
    Quant quant;

    // Search parameters
    int32_t ef = 32;

    // Memory prefetch parameters
    int32_t po = 1;
    int32_t pl = 1;
    int32_t graph_po = 1;

    // Optimization parameters
    constexpr static int32_t kOptimizePoints = 300000;
    constexpr static int32_t kTryPos = 10;
    constexpr static int32_t kTryPls = 10;
    constexpr static int32_t kTryK = 10;
    int32_t sample_points_num;
    std::vector<float> optimize_queries;

    GraphSearcher(Graph<int32_t> g)
        : graph(std::move(g)), graph_po(graph.K / 16) {}

    void SetData(const float *data, int32_t n, int32_t dim) override
    {
      this->nb = n;
      this->d = dim;
      quant = Quant(d);
      // printf("Starting quantizer training\n");
      auto t1 = std::chrono::high_resolution_clock::now();
      quant.train(data, n);
      quant.add(data, n);
      auto t2 = std::chrono::high_resolution_clock::now();
      // printf("Done quantizer training, cost %.2lfs\n",
      //        std::chrono::duration<double>(t2 - t1).count());

      sample_points_num = std::min(kOptimizePoints, nb - 1);
      std::vector<int32_t> sample_points(sample_points_num);
      std::mt19937 rng;
      GenRandom(rng, sample_points.data(), sample_points_num, nb);
      optimize_queries.resize((int64_t)sample_points_num * d);
      for (int32_t i = 0; i < sample_points_num; ++i)
      {
        memcpy(optimize_queries.data() + (int64_t)i * d,
               data + (int64_t)sample_points[i] * d, d * sizeof(float));
      }

      // Optimize();
    }

    void SetEf(int32_t ef) override { this->ef = ef; }

    int32_t GetEf() const override { return ef; }

    void Optimize(int32_t = 0) override
    {
      std::vector<int32_t> try_pos(std::min(kTryPos, graph.K));
      std::vector<int32_t> try_pls(
          std::min(kTryPls, (int32_t)upper_div(quant.code_size(), 64)));
      std::iota(try_pos.begin(), try_pos.end(), 1);
      std::iota(try_pls.begin(), try_pls.end(), 1);
      std::vector<int32_t> dummy_dst(kTryK);

      auto f = [&]
      {
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = 0; i < sample_points_num; ++i)
        {
          // printf("%d  %d\n", i, sample_points_num);
          Search(optimize_queries.data() + (int64_t)i * d, kTryK,
                 dummy_dst.data());
        }
      };
      printf("=============Start optimization=============\n");
      // warmup
      f();
      float min_ela = std::numeric_limits<float>::max();
      int32_t best_po = 0, best_pl = 0;
      for (auto try_po : try_pos)
      {
        for (auto try_pl : try_pls)
        {
          this->po = try_po;
          this->pl = try_pl;
          auto st = std::chrono::high_resolution_clock::now();
          f();
          auto ed = std::chrono::high_resolution_clock::now();
          auto ela = std::chrono::duration<double>(ed - st).count();
          if (ela < min_ela)
          {
            min_ela = ela;
            best_po = try_po;
            best_pl = try_pl;
          }
        }
      }
      float baseline_ela;
      {
        this->po = 1;
        this->pl = 1;
        auto st = std::chrono::high_resolution_clock::now();
        f();
        auto ed = std::chrono::high_resolution_clock::now();
        baseline_ela = std::chrono::duration<double>(ed - st).count();
      }
      float slow_ela;
      {
        this->po = 0;
        this->pl = 0;
        auto st = std::chrono::high_resolution_clock::now();
        f();
        auto ed = std::chrono::high_resolution_clock::now();
        slow_ela = std::chrono::duration<double>(ed - st).count();
      }

      printf("settint best po = %d, best pl = %d\n"
             "gaining %6.2f%% performance improvement wrt baseline\ngaining "
             "%6.2f%% performance improvement wrt slow\n============="
             "Done optimization=============\n",
             best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1),
             100.0 * (slow_ela / min_ela - 1));
      this->po = best_po;
      this->pl = best_pl;
      std::vector<float>().swap(optimize_queries);
    }

    void Search(const float *q, int32_t k, int32_t *ids,
                float *dis = nullptr) const override
    {
      auto computer = quant.get_computer(q);
      inference::LinearPool<typename Quant::ComputerType::dist_type> pool(
          nb, std::max(k, ef), k);
      graph.initialize_search(pool, computer);
      SearchImpl(pool, computer);
      for (int32_t i = 0; i < k; ++i)
      {
        ids[i] = pool.id(i);
        if (dis != nullptr)
        {
          dis[i] = pool.dist(i);
        }
      }
    }

    void SearchBatch(const float *q, int32_t nq, int32_t k, int32_t *ids,
                     float *dis = nullptr) const override
    {
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < nq; ++i)
      {
        Search(q + i * d, k, ids + i * k, dis ? dis + i * k : nullptr);
      }
    }

    void SearchImpl(inference::NeighborPoolConcept auto &pool,
                    ComputerConcept auto &computer) const
    {
      alignas(64) int32_t edge_buf[graph.K];
      while (pool.has_next())
      {
        auto u = pool.pop();
        graph.prefetch(u, graph_po);
        int32_t edge_size = 0;
        for (int32_t i = 0; i < graph.K; ++i)
        {
          int32_t v = graph.at(u, i);
          if (v == -1)
          {
            break;
          }
          if (pool.is_visited(v))
          {
            continue;
          }
          pool.set_visited(v);
          edge_buf[edge_size++] = v;
        }
        for (int i = 0; i < std::min(po, edge_size); ++i)
        {
          computer.prefetch(edge_buf[i], pl);
        }
        for (int i = 0; i < edge_size; ++i)
        {
          if (i + po < edge_size)
          {
            computer.prefetch(edge_buf[i + po], pl);
          }
          auto v = edge_buf[i];
          auto cur_dist = computer(v);
          pool.insert(v, cur_dist);
        }
      }
    }
  };

  inline std::unique_ptr<GraphSearcherBase>
  create_searcher(Graph<int32_t> graph, const std::string &metric,
                  const std::string &quantizer = "FP16")
  {
    using RType = std::unique_ptr<GraphSearcherBase>;
    auto m = metric_map[metric];
    auto qua = quantizer_map[quantizer];

    if (qua == QuantizerType::PQ8) {
      if( m == Metric::IP) {
        RType ret = std::make_unique<GraphSearcher<ProductQuant<Metric::IP>>>(std::move(graph));
        if(params::PQ8_REFINE) {
          ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(std::move(ret), params::PQ8_REFINE_FACTOR);
        }
        return ret;
      } else {
        printf("Metric not suppported\n");
        return nullptr;
      }
    }
    if (qua == QuantizerType::SQ8U)
    {
      if (m == Metric::IP)
      {
        RType ret =
            std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::IP>>>(
                std::move(graph));
        if (params::SQ8U_REFINE)
        {
          ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
              std::move(ret), params::SQ8U_REFINE_FACTOR);
        }
        return ret;
      }
      else if (m == Metric::L2)
      {
        RType ret =
            std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::L2>>>(
                std::move(graph));
        if (params::SQ8U_REFINE)
        {
          ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
              std::move(ret), params::SQ8U_REFINE_FACTOR);
        }
        return ret;
      }
      else
      {
        printf("Metric not suppported\n");
        return nullptr;
      }
    }
    else if (qua == QuantizerType::SQ4U)
    {
      if (m == Metric::IP)
      {
        RType ret =
            std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::IP>>>(
                std::move(graph));
        if (params::SQ4U_REFINE)
        {
          ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
              std::move(ret), params::SQ4U_REFINE_FACTOR);
        }
        return ret;
      }
      else if (m == Metric::L2)
      {
        RType ret =
            std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::L2>>>(
                std::move(graph));
        if (params::SQ4U_REFINE)
        {
          ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
              std::move(ret), params::SQ4U_REFINE_FACTOR);
        }
        return ret;
      }
      else
      {
        printf("Metric not suppported\n");
        return nullptr;
      }
    }
    else if (qua == QuantizerType::FP32)
    {
      if (m == Metric::IP)
      {
        RType ret = std::make_unique<GraphSearcher<FP32Quantizer<Metric::IP>>>(
            std::move(graph));
        return ret;
      }
      else
      {
        printf("Metric not suppported\n");
        return nullptr;
      }
    }
    else if (qua == QuantizerType::FP16)
    {
      if (m == Metric::IP)
      {
        RType ret = std::make_unique<GraphSearcher<FP16Quantizer<Metric::IP>>>(
            std::move(graph));
        return ret;
      }
      else if (m == Metric::L2) 
      {
        RType ret = std::make_unique<GraphSearcher<FP16Quantizer<Metric::L2>>>(
            std::move(graph));
        return ret;
      } else 
      {
        printf("Metric not suppported\n");
        return nullptr;
      }
    }
    else if (qua == QuantizerType::SQ6)
    {
      RType ret = std::make_unique<GraphSearcher<SQ6Quantizer<Metric::IP>>>(
          std::move(graph));
      return ret;
    }
    else if (qua == QuantizerType::SQ8P)
    {
      if (m == Metric::IP)
      {
        RType ret = std::make_unique<GraphSearcher<SQ8Quantizer2<Metric::IP>>>(
            std::move(graph));
        if(params::SQ8P_REFINE)
        {
          ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
              std::move(ret), params::SQ8P_REFINE_FACTOR);
        }
        return ret;
      }
      else if (m == Metric::L2)
      {
        RType ret = std::make_unique<GraphSearcher<SQ8Quantizer2<Metric::L2>>>(
            std::move(graph));
        return ret;
      }
      else
      {
        printf("Metric not suppported\n");
        return nullptr;
      }
    }
    else
    {
      printf("Quantizer type not supported\n");
      return nullptr;
    }
  }

} // namespace ann
