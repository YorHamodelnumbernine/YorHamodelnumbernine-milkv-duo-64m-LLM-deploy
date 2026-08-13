SDK_ROOT := $(HOME)/Documents/MilkV_duo_project/cvitek-tdl-sdk-cv180x
MW_PATH  := $(SDK_ROOT)/sample/3rd/middleware/v2
TPU_PATH := $(SDK_ROOT)/sample/3rd/tpu
IVE_PATH := $(SDK_ROOT)/sample/3rd/ive

CROSS_COMPILE := riscv64-unknown-linux-musl-
CC  := $(CROSS_COMPILE)gcc
CXX := $(CROSS_COMPILE)g++

ARCH := riscv64
CFLAGS := -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d
CFLAGS += -O3 -DNDEBUG -D_MIDDLEWARE_V2_ -DCV180X -DUSE_TPU_IVE
CFLAGS += -std=gnu11 -Wno-pointer-to-int-cast -fsigned-char
CFLAGS += -I$(SDK_ROOT)/include
CFLAGS += -I$(TPU_PATH)/include
CFLAGS += -I$(TPU_PATH)/include/cvikernel
CFLAGS += -I$(TPU_PATH)/include/bmkernel/bm1822
CFLAGS += -I./common

LDFLAGS := -L$(TPU_PATH)/lib -lcnpy -lcvikernel -lcvimath -lcviruntime -lz -lm
LDFLAGS += -lpthread -latomic
LDFLAGS += -s

# --- Find all test source files ---
ELEMWISE_SRCS := $(wildcard 03_elemwise/*.c)
LOGIC_SRCS    := $(wildcard 04_logic/*.c)
SHIFT_SRCS    := $(wildcard 05_shift_copy/*.c)
LUT_SRCS      := $(wildcard 06_lookup/*.c)
POOL_SRCS     := $(wildcard 07_pooling/*.c)
CONV_SRCS     := $(wildcard 01_conv/*.c)
MATMUL_SRCS   := $(wildcard 02_matmul/*.c)

ROOT_SRCS := $(wildcard *.c)
ALL_SRCS   := $(ELEMWISE_SRCS) $(LOGIC_SRCS) $(SHIFT_SRCS) $(LUT_SRCS) \
              $(POOL_SRCS) $(CONV_SRCS) $(MATMUL_SRCS) $(ROOT_SRCS)

BINS := $(ALL_SRCS:.c=)

.PHONY: all clean

all: $(BINS)

# gen_rand_weights: pure C, no TPU deps, runs on Duo
gen_rand_weights: gen_rand_weights.c
	$(CC) $(CFLAGS) -o $@ $< -lm

smollm2_demo: smollm2_demo.c common/tpu_bench.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# smollm2_pool_demo: INT8 default + WT_INT4=1 runtime switch (Design A)
smollm2_pool_demo: smollm2_pool_demo.c common/tpu_bench.h int4_common.c int4_common.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ smollm2_pool_demo.c int4_common.c

# smollm2_pool_b2_i4: alias binary for CEO regression command
smollm2_pool_b2_i4: smollm2_pool_demo.c common/tpu_bench.h int4_common.c int4_common.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ smollm2_pool_demo.c int4_common.c

%: %.c common/tpu_bench.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
