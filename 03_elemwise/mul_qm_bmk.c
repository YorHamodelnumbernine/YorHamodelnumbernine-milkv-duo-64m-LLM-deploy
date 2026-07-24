/* TPU mul (quantized) benchmark — bmk1822 API (element_wise_mac).
   res = a * b + res_init.  Preload res with 0, b=multiplier, rshift for quant.
   a=[1..8], b_const=4, rshift=2 => (a*4)>>2 = a */
#include "../common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define N 8

int main() {
  int errs = 0;

  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { fprintf(stderr,"RT_Init fail\n"); return 1; }

  CVI_RT_MEM neuron_mem = CVI_RT_MemAlloc(rt, 4096);
  if (!neuron_mem) { fprintf(stderr,"MemAlloc fail\n"); CVI_RT_DeInit(rt); return 1; }
  uint64_t neuron_pa = CVI_RT_MemGetPAddr(neuron_mem);
  uint8_t *neuron_va = CVI_RT_MemGetVAddr(neuron_mem);
  CVI_RT_SetBaseReg(rt, 0, neuron_pa);

  static const int8_t a_data[N] = {1,2,3,4,5,6,7,8};
  static const int8_t zero[N]   = {0,0,0,0,0,0,0,0};
  static const int8_t exp_lo[N] = {1,2,3,4,5,6,7,8};
  static const int8_t exp_hi[N] = {0,0,0,0,0,0,0,0};

  memcpy(neuron_va, a_data, N);
  memcpy(neuron_va+16, zero, N);  /* pre-init res_low  area with zeros */
  memcpy(neuron_va+32, zero, N);  /* pre-init res_high area with zeros */
  CVI_RT_MemFlush(rt, neuron_mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { fprintf(stderr,"bmk reg fail\n"); goto out_mem; }

  bmk1822_tensor_lmem_shape_t s8 = {1,1,1,N};
  bmk1822_tensor_lmem_t *tl_a  = bmk1822_lmem_alloc_tensor(bmk, s8, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_rl = bmk1822_lmem_alloc_tensor(bmk, s8, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_rh = bmk1822_lmem_alloc_tensor(bmk, s8, FMT_I8, 1);
  if (!tl_a||!tl_rl||!tl_rh) { fprintf(stderr,"alloc fail\n"); goto out_bmk; }

  bmk1822_tensor_tgmem_t tg_a = {
    .base_reg_index = 0, .start_address = 0, .fmt = FMT_I8, .shape = {1,1,1,N},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,N}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_rl = {
    .base_reg_index = 0, .start_address = 16, .fmt = FMT_I8, .shape = {1,1,1,N},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,N}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_rh = {
    .base_reg_index = 0, .start_address = 32, .fmt = FMT_I8, .shape = {1,1,1,N},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,N}, FMT_I8),
  };

  /* Load a, and pre-init result tensors with 0 */
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_a, tl_a});
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_rl, tl_rl});
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_rh, tl_rh});

  /* MAC: res_16bit = a * b + res_init.
     b_const=4, rshift=2 => (a*4)>>2 = a.
     res_init is 0, so result is pure multiply+quantize. */
  bmk1822_tiu_element_wise_mac_param_t p = {
    .res_high = tl_rh,
    .res_low = tl_rl,
    .a = tl_a,
    .b_is_const = 1,
    .b_const = { .val = 4, .is_signed = 1 },
    .res_is_int8 = 0,   /* keep 16-bit output, split hi/lo */
    .relu_enable = 0,
    .lshift_bits = 0,
    .rshift_bits = 2,
    .layer_id = 1,
  };
  if (!bmk1822_tiu_element_wise_mac(bmk, &p)) { fprintf(stderr,"mac rejected\n"); goto out_bmk; }

  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_rl, &tg_rl});
  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_rh, &tg_rh});

  uint32_t cmd_sz;
  uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size;
  bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);

  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  if (!dmabuf_mem) { fprintf(stderr,"dmabuf alloc fail\n"); goto out_bmk; }
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, neuron_pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  CVI_RT_MEM loaded;
  int rc = CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize + pmu_size, neuron_pa, 0, false, &loaded);
  if (rc != 0) { fprintf(stderr,"Load fail rc=%d\n", rc); goto out_dmabuf; }
  CVI_RT_ARRAYBASE arr = { .gaddr_base0 = neuron_pa };
  rc = CVI_RT_RunCmdbufEx(rt, loaded, &arr);
  if (rc != 0) { fprintf(stderr,"Run fail rc=%d\n", rc); goto out_dmabuf; }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  int64_t t_ns = (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);

  CVI_RT_MemInvld(rt, neuron_mem);
  int8_t *v = (int8_t*)neuron_va;
  errs = 0;
  for (int i = 0; i < N; i++) { if (v[16+i] != exp_lo[i]) errs++; }
  for (int i = 0; i < N; i++) { if (v[32+i] != exp_hi[i]) errs++; }
  printf("  [mul_qm_bmk] lo=[%d,%d,%d,%d,%d,%d,%d,%d] hi=[%d,%d,%d,%d,%d,%d,%d,%d]  exp_lo=[%d,%d,%d,%d,%d,%d,%d,%d]  %s  time: %.2f us\n",
    v[16],v[17],v[18],v[19],v[20],v[21],v[22],v[23],
    v[32],v[33],v[34],v[35],v[36],v[37],v[38],v[39],
    exp_lo[0],exp_lo[1],exp_lo[2],exp_lo[3],exp_lo[4],exp_lo[5],exp_lo[6],exp_lo[7],
    errs?"FAIL":"OK", t_ns/1000.0);

out_dmabuf:
  CVI_RT_MemFree(rt, dmabuf_mem);
out_bmk:
  bmk1822_cleanup(bmk);
out_mem:
  CVI_RT_MemFree(rt, neuron_mem);
  CVI_RT_DeInit(rt);
  return errs?1:0;
}
