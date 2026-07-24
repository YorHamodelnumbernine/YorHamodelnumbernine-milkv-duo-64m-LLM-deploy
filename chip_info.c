/* Dump CV180X chip info from cvikernel context */
#include "../common/tpu_bench.h"
int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_chip_info_t *i = &ctx.cvk_ctx->info;
  printf("version:     %u\n", i->version);
  printf("node_num:    %u\n", i->node_num);
  printf("node_shift:  %u\n", i->node_shift);
  printf("npu_num:     %u\n", i->npu_num);
  printf("npu_shift:   %u\n", i->npu_shift);
  printf("eu_num:      %u\n", i->eu_num);
  printf("eu_shift:    %u\n", i->eu_shift);
  printf("lmem_size:   %u (0x%x)\n", i->lmem_size, i->lmem_size);
  printf("lmem_shift:  %u\n", i->lmem_shift);
  printf("lmem_banks:  %u\n", i->lmem_banks);
  printf("lmem_bank_size: %u\n", i->lmem_bank_size);
  printf("lmem_start:  0x%llx\n", (unsigned long long)i->lmem_start);
  printf("gmem_start:  0x%llx\n", (unsigned long long)i->gmem_start);
  printf("gmem_size:   0x%llx\n", (unsigned long long)i->gmem_size);
  printf("features:    0x%llx\n", (unsigned long long)i->features);
  tpu_cleanup(&ctx);
  return 0;
}
