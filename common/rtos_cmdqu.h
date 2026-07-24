/* Userspace header for cvi-rtos-cmdqu mailbox interface.
   Derived from middleware/v2/include/rtos_cmdqu.h (stripped of kernel deps).
*/
#ifndef RTOS_CMDQU_USER_H
#define RTOS_CMDQU_USER_H

#include <stdint.h>
#include <sys/ioctl.h>

/* IP types matching the kernel driver enum */
enum IP_TYPE {
    IP_ISP    = 0,
    IP_VCODEC = 1,
    IP_VIP    = 2,
    IP_VI     = 3,
    IP_RGN    = 4,
    IP_AUDIO  = 5,
    IP_SYSTEM = 6,
    IP_CAMERA = 7,
    IP_LIMIT  = 8,
};

/* System command IDs (must match FreeRTOS rtos_cmdqu.h) */
enum SYS_CMD_ID {
    SYS_CMD_INFO_TRANS              = 0x50,
    SYS_CMD_INFO_LINUX_INIT_DONE,
    SYS_CMD_INFO_RTOS_INIT_DONE,
    SYS_CMD_INFO_STOP_ISR,
    SYS_CMD_INFO_STOP_ISR_DONE,
    SYS_CMD_INFO_LINUX,
    SYS_CMD_INFO_RTOS,
    SYS_CMD_SYNC_TIME,
    SYS_CMD_INFO_DUMP_MSG,
    SYS_CMD_INFO_DUMP_EN,
    SYS_CMD_INFO_DUMP_DIS,
    SYS_CMD_INFO_TRACE_SNAPSHOT_START,
    SYS_CMD_INFO_TRACE_SNAPSHOT_STOP,
    SYS_CMD_INFO_TRACE_STREAM_START,
    SYS_CMD_INFO_TRACE_STREAM_STOP,
    SYS_CMD_INFO_LIMIT,
};

struct valid_t {
    uint8_t linux_valid;
    uint8_t rtos_valid;
} __attribute__((packed));

typedef union resv_t {
    struct valid_t valid;
    uint16_t mstime; /* 0: noblock, 0xFFFF: block infinite */
} resv_t;

typedef struct cmdqu_t cmdqu_t;
/* cmdqu size must be 8 bytes (mailbox buffer slot size) */
struct cmdqu_t {
    uint8_t ip_id;
    uint8_t cmd_id : 7;
    uint8_t block  : 1;
    union resv_t resv;
    uint32_t param_ptr;
} __attribute__((packed)) __attribute__((aligned(0x8)));

/* ioctl commands (must match kernel driver) */
#define RTOS_CMDQU_SEND         _IOW('r', 1, unsigned long)
#define RTOS_CMDQU_SEND_WAIT    _IOW('r', 2, unsigned long)
#define RTOS_CMDQU_SEND_WAKEUP  _IOW('r', 3, unsigned long)

#endif /* RTOS_CMDQU_USER_H */
