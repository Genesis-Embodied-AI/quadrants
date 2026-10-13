/* CPU reader that never calls Metal. Times two back-to-back walks of one fixed 64 MiB buffer.
   The heap slowdown makes the first walk slow while this process stays alive. The second walk stays fast.
   Process exit of the Metal workers brings the first walk back down. */
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _POSIX_C_SOURCE 200809L
#endif
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>

#define NODES (1u << 20)
#define VISITS (1u << 18)

typedef struct Node {
    struct Node *next;
    unsigned char pad[64 - sizeof(void *)];
} Node;

static volatile sig_atomic_t running = 1;

static void stop(int signum) {
    (void)signum;
    running = 0;
}

static double seconds(clockid_t clock) {
    struct timespec ts;
    clock_gettime(clock, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static Node *walk(Node *node) {
    for (uint32_t i = 0; i < VISITS; i++) node = node->next;
    return node;
}

int main(void) {
    long page = sysconf(_SC_PAGESIZE);
    Node *nodes = calloc(NODES, sizeof(Node));
    uint32_t *order = calloc(NODES, sizeof(uint32_t));
    if (!nodes || !order || page <= 0) return 1;
    for (uint32_t i = 0; i < NODES; i++) order[i] = i;
    uint64_t state = 42;
    for (uint32_t i = NODES - 1; i > 0; i--) {
        state = state * UINT64_C(6364136223846793005) + 1;
        uint32_t j = (uint32_t)((state >> 32) % (i + 1));
        uint32_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
    for (uint32_t i = 0; i < NODES; i++) nodes[order[i]].next = &nodes[order[(i + 1) % NODES]];
    Node *head = &nodes[order[0]];
    Node *expected = &nodes[order[VISITS]];
    free(order);

    /* Fault the pages in once. Later samples are re-reads of memory this process already owns. */
    if (walk(head) != expected) return 2;

    signal(SIGTERM, stop);
    signal(SIGINT, stop);
    printf("cpu_reader pid=%d bytes=%zu page=%ld\n", getpid(), (size_t)NODES * sizeof(Node), page);
    fflush(stdout);

    for (int sample = 0; running; sample++) {
        struct rusage before, after;
        if (getrusage(RUSAGE_SELF, &before)) return 3;
        double w0 = seconds(CLOCK_MONOTONIC);
        double c0 = seconds(CLOCK_THREAD_CPUTIME_ID);
        Node *got = walk(head);
        double cpu1 = seconds(CLOCK_THREAD_CPUTIME_ID) - c0;
        double wall1 = seconds(CLOCK_MONOTONIC) - w0;
        c0 = seconds(CLOCK_THREAD_CPUTIME_ID);
        w0 = seconds(CLOCK_MONOTONIC);
        Node *got2 = walk(head);
        double cpu2 = seconds(CLOCK_THREAD_CPUTIME_ID) - c0;
        double wall2 = seconds(CLOCK_MONOTONIC) - w0;
        if (getrusage(RUSAGE_SELF, &after)) return 3;
        if (got != expected || got2 != expected) return 4;

        size_t pages = 0, resident = 0;
#ifdef __APPLE__
        uintptr_t begin = (uintptr_t)nodes & ~(uintptr_t)(page - 1);
        uintptr_t end = ((uintptr_t)nodes + (size_t)NODES * sizeof(Node) + (size_t)page - 1) & ~(uintptr_t)(page - 1);
        pages = (end - begin) / (size_t)page;
        unsigned char *vec = calloc(pages, 1);
        if (vec && mincore((void *)begin, end - begin, vec) == 0) {
            for (size_t i = 0; i < pages; i++) resident += (vec[i] & MINCORE_INCORE) != 0;
        }
        free(vec);
#endif
        printf("sample=%d epoch=%.3f wall1=%.6f cpu1=%.6f wall2=%.6f cpu2=%.6f minor=%ld major=%ld resident=%zu pages=%zu\n",
               sample, seconds(CLOCK_REALTIME), wall1, cpu1, wall2, cpu2, after.ru_minflt - before.ru_minflt,
               after.ru_majflt - before.ru_majflt, resident, pages);
        fflush(stdout);
        sleep(5);
    }
    printf("cpu_reader exit\n");
    return 0;
}
