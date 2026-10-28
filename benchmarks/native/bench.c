/* Hand-written sequential baselines for the Appendix B loops. Reads data.txt
   (n, then n deltas, then n text bytes) and prints "name best_ns checksum". */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef long long i64;
static long n; static i64 *xs; static unsigned char *text;
static volatile i64 sink;

static i64 now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (i64)t.tv_sec * 1000000000LL + t.tv_nsec; }
static i64 max2(i64 a, i64 b){ return a > b ? a : b; }

static i64 *prefix_sum(void){ i64 *out = malloc(sizeof(i64) * n), s = 0; for(long i = 0; i < n; i++){ s += xs[i]; out[i] = s; } return out; }
static i64 *balances(const i64 *rate, const i64 *dep){ i64 *out = malloc(sizeof(i64) * n), b = 0; for(long i = 0; i < n; i++){ b = b * rate[i] + dep[i]; out[i] = b; } return out; }
static i64 *stock_levels(i64 cap){ i64 *out = malloc(sizeof(i64) * n), s = 0; for(long i = 0; i < n; i++){ s += xs[i]; s = s < 0 ? 0 : (s > cap ? cap : s); out[i] = s; } return out; }
static i64 word_count(void){ int in_word = 0; i64 c = 0; for(long i = 0; i < n; i++){ unsigned char ch = text[i]; int alpha = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); if(alpha && !in_word) c++; in_word = alpha; } return c; }
static void viterbi(i64 *ra, i64 *rb){ i64 a = 0, b = 0; for(long i = 0; i < n; i++){ i64 x = xs[i], na = max2(a + x, b - x), nb = max2(a - x, b + x); a = na; b = nb; } *ra = a; *rb = b; }

#define REPS 200
#define SCAN(name, call) { i64 best = -1, sum = 0, last = 0; for(int r = 0; r < REPS; r++){ i64 t0 = now(); i64 *o = call; i64 dt = now() - t0; if(best < 0 || dt < best) best = dt; sum = 0; for(long i = 0; i < n; i++) sum += o[i]; last = o[n - 1]; free(o); } printf("%s %lld %lld:%lld\n", name, best, sum, last); }

int main(void){
  FILE *f = fopen("data.txt", "r"); if(!f) return 1;
  if(fscanf(f, "%ld", &n) != 1) return 1;
  xs = malloc(sizeof(i64) * n); text = malloc(n);
  for(long i = 0; i < n; i++) if(fscanf(f, "%lld", &xs[i]) != 1) return 1;
  for(long i = 0; i < n; i++){ int c; if(fscanf(f, "%d", &c) != 1) return 1; text[i] = (unsigned char)c; }
  i64 *rate = malloc(sizeof(i64) * n); for(long i = 0; i < n; i++) rate[i] = 1;

  SCAN("prefix_sum", prefix_sum())
  SCAN("linear_recurrence", balances(rate, xs))
  SCAN("saturating_counters", stock_levels(50))
  { i64 best = -1, c = 0; for(int r = 0; r < REPS; r++){ i64 t0 = now(); c = word_count(); i64 dt = now() - t0; sink = c; if(best < 0 || dt < best) best = dt; } printf("tokenizer %lld %lld\n", best, c); }
  { i64 best = -1, a = 0, b = 0; for(int r = 0; r < REPS; r++){ i64 t0 = now(); viterbi(&a, &b); i64 dt = now() - t0; sink = a + b; if(best < 0 || dt < best) best = dt; } printf("viterbi %lld %lld:%lld\n", best, a, b); }
  return 0;
}
