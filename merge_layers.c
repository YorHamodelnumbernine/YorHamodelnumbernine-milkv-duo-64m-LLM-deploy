#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define L 30
#define BUF_SZ (4*1024*1024)

static char *file_names[] = {
    "rms_attn.f32",
    "Wq.i8","Wk.i8","Wv.i8","Wo.i8",
    "rms_ffn.f32",
    "ffn_up.i8","ffn_gate.i8","ffn_down.i8",
    NULL
};

static char buf[BUF_SZ];

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <weight_dir> <out_dir>\n", argv[0]);
        return 1;
    }
    mkdir(argv[2], 0777);
    for (int l = 0; l < L; l++) {
        char out_path[256];
        snprintf(out_path, sizeof(out_path), "%s/layer%d.bin", argv[2], l);
        int out_fd = creat(out_path, 0666);
        if (out_fd < 0) { perror(out_path); return 1; }
        int total = 0;
        for (int i = 0; file_names[i]; i++) {
            char in_path[256];
            snprintf(in_path, sizeof(in_path), "%s/layer%d_%s", argv[1], l, file_names[i]);
            int in_fd = open(in_path, O_RDONLY);
            if (in_fd < 0) { perror(in_path); close(out_fd); return 1; }
            while (1) {
                int n = read(in_fd, buf, BUF_SZ);
                if (n <= 0) break;
                write(out_fd, buf, n);
                total += n;
            }
            close(in_fd);
        }
        close(out_fd);
        printf("layer%d.bin: %d bytes\n", l, total);
    }
    printf("Done.\n");
    return 0;
}
