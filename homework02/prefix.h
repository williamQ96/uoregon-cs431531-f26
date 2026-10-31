#ifndef PREFIX_H
#define PREFIX_H
#ifdef __cplusplus
extern "C" {
#endif
/* Inclusive scans, with sums representable as int. n <= 0 does nothing.
 * Arrays must not overlap. p1 uses src as scratch and may overwrite it;
 * p2 and the serial baseline preserve src. */
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);
#ifdef __cplusplus
}
#endif
#endif
