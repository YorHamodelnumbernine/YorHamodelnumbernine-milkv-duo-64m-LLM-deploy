/* abort_catch_probe.c — 验证 sigsetjmp/siglongjmp 捕获 SIGABRT 在
 * musl/RISC-V 上可用 (引擎 rt_alloc_safe 的机制同款). 每次循环 raise(SIGABRT),
 * 若 handler 正常 longjmp 则继续; 若机制失效则进程死在 SIGABRT (RC=134).
 * 用法: abort_catch_probe [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <setjmp.h>

static sigjmp_buf jb;
static volatile sig_atomic_t armed = 0;

static void h(int sig) {
    if (armed) siglongjmp(jb, 1);
    signal(sig, SIG_DFL);
    raise(sig);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 3;
    struct sigaction sa;
    sa.sa_handler = h;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGABRT, &sa, NULL);
    for (int i = 0; i < iters; i++) {
        armed = 1;
        if (sigsetjmp(jb, 1) == 0) {
            raise(SIGABRT);
            fprintf(stderr, "iter %d: NOT longjmp'd (bad)\n", i);
            return 1;
        }
        armed = 0;
        printf("iter %d: SIGABRT CAUGHT, continue OK\n", i);
        fflush(stdout);
    }
    printf("all %d iters OK (siglongjmp works on this toolchain)\n", iters);
    return 0;
}
