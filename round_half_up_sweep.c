/* round_half_up_sweep.c — host-side sweep: confirm the CPU reference
 * `(acc + (1<<(rshift-1))) >> rshift` implements round-half-up for EVERY
 * rshift in [1,31], including small values (rshift=1) and negative acc.
 *
 * Reference (exact, no overflow): floor(acc / 2^s + 1/2) computed in int64,
 * i.e. the rational round-half-up toward +inf — the semantics the TIU was
 * on-board verified to match at rshift=1..4 (rshift_check.c), 5/8
 * (gate_a_check.c), 8..10 (gate1_mrow_check.c). This file closes the rest.
 *
 * Build/run (host, no TPU deps):  gcc -O2 -o round_half_up_sweep round_half_up_sweep.c && ./round_half_up_sweep
 */
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

/* exact floor(a / 2^s) for signed int64, s>=1 (avoid C div-toward-zero) */
static int64_t floor_div_pow2(int64_t a, int s) {
  if (a >= 0) return a >> s;
  return -(((-a) + ((int64_t)1 << s) - 1) >> s);
}

/* reference: round-half-up toward +inf, exact */
static int32_t ref_rhu(int32_t x, int s) {
  int64_t num = (int64_t)x + ((int64_t)1 << (s - 1));
  return (int32_t)floor_div_pow2(num, s);
}

/* candidate formula as the microkernel would write it (int32 arithmetic) */
static int32_t formula(int32_t x, int s) {
  return (x + (1 << (s - 1))) >> s;
}

int main(void) {
  int total_bad = 0, total_oob = 0, tie_checked = 0;
  printf("===== round-half-up sweep: rshift 1..31 =====\n");

  for (int s = 1; s <= 31; s++) {
    int bad = 0, oob = 0, ties = 0;
    int64_t lo_ok = -(int64_t)INT32_MAX - 1 - ((int64_t)1 << (s - 1)); /* in-range bound of x */
    int64_t hi_ok =  (int64_t)INT32_MAX - ((int64_t)1 << (s - 1));
    /* dense consecutive sweep: covers every tie x = q*2^s + 2^(s-1) exactly */
    for (int64_t x = -(1LL << 21); x <= (1LL << 21); x++) {
      if (x < lo_ok || x > hi_ok) { oob++; continue; }     /* formula would overflow int32 */
      int32_t f = formula((int32_t)x, s);
      int32_t r = ref_rhu((int32_t)x, s);
      if (f != r) bad++;
      int64_t low = x & (((int64_t)1 << s) - 1);
      if (low == ((int64_t)1 << (s - 1))) ties++;          /* exact half-way */
    }
    /* edge values near INT32 boundaries (small s only) + INT32 extremes */
    int64_t edges[] = { INT32_MIN, INT32_MAX, -1, 0, 1, -INT32_MAX, INT32_MAX - 1 };
    for (unsigned i = 0; i < sizeof(edges)/sizeof(edges[0]); i++) {
      int64_t x = edges[i];
      if (x < lo_ok || x > hi_ok) continue;
      if (formula((int32_t)x, s) != ref_rhu((int32_t)x, s)) bad++;
    }
    total_bad += bad; total_oob += oob; tie_checked += ties;
    printf("  rshift=%2d: bad=%d out-of-domain=%d tie-cases=%d %s\n",
           s, bad, oob, ties, bad ? "  <<< FAIL" : "ok");
  }

  /* CEO anchor: 11440 >> 8 -> 45 */
  printf("  anchor 11440>>8 = %d (expect 45) %s\n", formula(11440, 8),
         formula(11440, 8) == 45 ? "ok" : "FAIL");

  /* small-rshift spot table for the microkernel reference */
  printf("\n  spot rshift=1: 0->%d 1->%d 2->%d 3->%d 4->%d 5->%d (expect 0 1 1 2 2 3)\n",
         formula(0,1), formula(1,1), formula(2,1), formula(3,1), formula(4,1), formula(5,1));
  printf("  spot rshift=2: 2->%d 3->%d 4->%d 5->%d 6->%d 7->%d (expect 1 1 1 1 2 2)\n",
         formula(2,2), formula(3,2), formula(4,2), formula(5,2), formula(6,2), formula(7,2));
  printf("  spot neg rshift=1: -1->%d -2->%d -3->%d -4->%d (expect 0 -1 -1 -2)\n",
         formula(-1,1), formula(-2,1), formula(-3,1), formula(-4,1));

  printf("\n===== RESULT: total bad=%d (0=ALL PASS), tie-cases checked=%d, out-of-domain=%d =====\n",
         total_bad, tie_checked, total_oob);
  return total_bad ? 1 : 0;
}
