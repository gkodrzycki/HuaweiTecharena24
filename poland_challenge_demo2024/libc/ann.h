#ifndef C_INCLUDE_ann_H_
#define C_INCLUDE_ann_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
void *ann_init(int K_features, int R, const char *metric);
void ann_free(void *ptr);
void set_ann_ef(void *ptr, int ann_ef);

void ann_add(void *ptr, int n, float * x, const char *store);
void ann_search(void *ptr, int n, const float* x, int k, float* distances, int32_t* labels, int num_p);

void ann_save(void *ptr, const char *path);
void ann_load(void *ptr, const char *path);



#ifdef __cplusplus
}
#endif
#endif