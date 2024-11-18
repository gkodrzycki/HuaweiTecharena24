#pragma once

#include "ann/quant/bf16_quant.hpp"
#include "ann/quant/calibrator.hpp"
#include "ann/quant/computer.hpp"
#include "ann/quant/quant_base.hpp"
#include "ann/quant/utils.hpp"
#include "ann/third_party/helpa/helpa/dot.hpp"

#include <atomic>
#include <cmath>

namespace ann {

template <Metric metric, typename Template = Quantizer<metric, 64, 8>>
struct SQ8Quantizer2 : Template {
  using type = SQ8Quantizer2;
  using data_type = int8_t;

  constexpr static int32_t ncount = 127;

  std::vector<bf16> maxs;
  std::vector<bf16> norms;

  SQ8Quantizer2() = default;

  explicit SQ8Quantizer2(int dim) : Template(dim) {}

  void train(const float *, int32_t) {}

  void add(const float *data, int32_t n) {
    this->storage.init(n);
    maxs.resize(n);
    if constexpr (metric == Metric::L2) {
      norms.resize(n);
    }
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
      const float *vec = data + (int64_t)i * this->dim();
      maxs[i] = bf16(encode(vec, (data_type *)this->get_code(i)));
      if constexpr (metric == Metric::L2) {
        norms[i] = bf16(-helpa::dot_fp32_fp32(vec, vec, this->dim()));
      }
    }
  }

  float encode(const float *from, data_type *to) const {
    float mx = 0.0f;
    for (int j = 0; j < this->dim(); ++j) {
      mx = std::max(mx, std::abs(from[j]));
    }
    for (int j = 0; j < this->dim(); ++j) {
      float x = from[j] / mx;
      x = limit_range_sym(x);
      int8_t y = x * ncount;
      to[j] = y;
    }
    return mx;
  }

  constexpr static auto dist_func =
      metric == Metric::L2
          ? [](const int8_t *x, const int8_t *y,
               const int32_t
                   d) { return helpa::l2_u8_s8((const uint8_t *)x, y, d); }
          : [](const int8_t *x, const int8_t *y, const int32_t d) {
              return helpa::dota_u8_s8((const uint8_t *)x, y, d);
            };

  constexpr static auto dist_func_sym = dist_func;

  using ComputerType =
      ComputerImpl<Tensor, dist_func, int32_t, float, int8_t, int8_t>;
  using SymComputerType =
      SymComputerImpl<Tensor, dist_func_sym, int32_t, int8_t>;

  auto get_computer(const float *query) const {
    return ComputerType(this->storage, query,
                        [this](const float *from, data_type *&to) {
                          to = (data_type *)align_alloc(this->code_size());
                          this->encode(from, to);
                        });
  }

  auto get_sym_computer() const { return SymComputerType(this->storage); }

};

} // namespace ann
