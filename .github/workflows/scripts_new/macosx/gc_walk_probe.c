/* Read-only CPython 3.10 GC-list diagnostic. Never writes interpreter object headers or releases the GIL. */
#define Py_BUILD_CORE
#include <Python.h>
#include <internal/pycore_gc.h>
#include <internal/pycore_interp.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#ifdef __APPLE__
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 10
#error This diagnostic is specific to CPython 3.10.
#endif

static double now(clockid_t clock) {
    struct timespec value;
    clock_gettime(clock, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static int address_order(const void *left, const void *right) {
    uintptr_t a = (uintptr_t)*(PyGC_Head *const *)left;
    uintptr_t b = (uintptr_t)*(PyGC_Head *const *)right;
    return (a > b) - (a < b);
}

static uint64_t arithmetic(void) {
    volatile uint64_t value = 1;
    for (int i = 0; i < 1000000; i++) value = value * UINT64_C(6364136223846793005) + 1;
    return value;
}

static int counters(uint64_t *cycles, uint64_t *instructions) {
    *cycles = *instructions = 0;
#ifdef __APPLE__
    struct rusage_info_v4 usage = {0};
    int status = proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&usage);
    if (!status) {
        *cycles = usage.ri_cycles;
        *instructions = usage.ri_instructions;
    }
    return status;
#else
    return -1;
#endif
}

static PyObject *cold(PyObject *self, PyObject *unused) {
    PyGC_Head *head = &PyThreadState_Get()->interp->gc.generations[2].head;
    double wall[3], cpu[3];
    uint64_t checksums[2];
    unsigned long long cycle_delta[3], instruction_delta[3];
    int counter_status[3];
    Py_ssize_t count = 0;
    for (int pass = 0; pass < 3; pass++) {
        uint64_t before_cycles, before_instructions, after_cycles, after_instructions;
        int before_status = counters(&before_cycles, &before_instructions);
        double w = now(CLOCK_MONOTONIC), c = now(CLOCK_THREAD_CPUTIME_ID);
        if (pass == 1) {
            PyGC_Head *node = _PyGCHead_NEXT(head);
            while (node != head && count < 10000000) {
                node = _PyGCHead_NEXT(node);
                count++;
            }
            if (node != head) {
                PyErr_SetString(PyExc_RuntimeError, "GC list exceeded diagnostic bound");
                return NULL;
            }
        } else {
            checksums[pass / 2] = arithmetic();
        }
        cpu[pass] = now(CLOCK_THREAD_CPUTIME_ID) - c;
        wall[pass] = now(CLOCK_MONOTONIC) - w;
        counter_status[pass] = counters(&after_cycles, &after_instructions) || before_status;
        cycle_delta[pass] = after_cycles - before_cycles;
        instruction_delta[pass] = after_instructions - before_instructions;
    }
    if (checksums[0] != checksums[1]) {
        PyErr_SetString(PyExc_RuntimeError, "Arithmetic control checksum mismatch");
        return NULL;
    }
    return Py_BuildValue("{s:n,s:(ddd),s:(ddd),s:(KKK),s:(KKK),s:(iii)}", "objects", count,
                         "wall_seconds", wall[0], wall[1], wall[2],
                         "thread_cpu_seconds", cpu[0], cpu[1], cpu[2],
                         "process_cycles", cycle_delta[0], cycle_delta[1], cycle_delta[2],
                         "process_instructions", instruction_delta[0], instruction_delta[1], instruction_delta[2],
                         "counter_status", counter_status[0], counter_status[1], counter_status[2]);
}

/* A fixed private graph independent of Python objects and Vulkan. Retained for this process lifetime. */
typedef struct MemoryNode {
    struct MemoryNode *volatile next;
    unsigned char padding[64 - sizeof(void *)];
} MemoryNode;
_Static_assert(sizeof(MemoryNode) == 64, "Control nodes must have a fixed stride");
static MemoryNode *memory_nodes = NULL, *memory_head = NULL, *memory_expected = NULL;

static PyObject *memory_control(PyObject *self, PyObject *unused) {
    const uint32_t count = 1u << 20, visits = 1u << 18;
    if (!memory_nodes) {
        MemoryNode *nodes = PyMem_RawMalloc((size_t)count * sizeof(MemoryNode));
        uint32_t *order = PyMem_RawMalloc((size_t)count * sizeof(uint32_t));
        if (!nodes || !order) {
            PyMem_RawFree(nodes);
            PyMem_RawFree(order);
            return PyErr_NoMemory();
        }
        for (uint32_t i = 0; i < count; i++) order[i] = i;
        uint64_t state = 42;
        for (uint32_t i = count - 1; i > 0; i--) {
            state = state * UINT64_C(6364136223846793005) + 1;
            uint32_t j = (uint32_t)(state >> 32) % (i + 1);
            uint32_t temporary = order[i];
            order[i] = order[j];
            order[j] = temporary;
        }
        for (uint32_t i = 0; i < count; i++) nodes[order[i]].next = &nodes[order[(i + 1) % count]];
        memory_nodes = nodes;
        memory_head = &nodes[order[0]];
        memory_expected = &nodes[order[visits]];
        PyMem_RawFree(order);
    }
    double wall[4], cpu[4];
    uint64_t checksums[2];
    for (int pass = 0; pass < 4; pass++) {
        double w = now(CLOCK_MONOTONIC), c = now(CLOCK_THREAD_CPUTIME_ID);
        if (pass == 0 || pass == 3) {
            checksums[pass / 3] = arithmetic();
        } else {
            MemoryNode *node = memory_head;
            for (uint32_t i = 0; i < visits; i++) node = node->next;
            if (node != memory_expected) {
                PyErr_SetString(PyExc_RuntimeError, "Fixed memory control checksum mismatch");
                return NULL;
            }
        }
        cpu[pass] = now(CLOCK_THREAD_CPUTIME_ID) - c;
        wall[pass] = now(CLOCK_MONOTONIC) - w;
    }
    if (checksums[0] != checksums[1]) {
        PyErr_SetString(PyExc_RuntimeError, "Memory-control arithmetic checksum mismatch");
        return NULL;
    }
    return Py_BuildValue("{s:i,s:i,s:i,s:(dddd),s:(dddd)}", "nodes", (int)count,
                         "visits", (int)visits, "bytes", (int)(count * sizeof(MemoryNode)),
                         "wall_seconds", wall[0], wall[1], wall[2], wall[3],
                         "thread_cpu_seconds", cpu[0], cpu[1], cpu[2], cpu[3]);
}

static PyObject *measure(PyObject *self, PyObject *objects) {
    if (!PyList_CheckExact(objects) || !PyList_GET_SIZE(objects)) {
        PyErr_SetString(PyExc_ValueError, "Expected a nonempty gc.get_objects(2) list");
        return NULL;
    }
    Py_ssize_t size = PyList_GET_SIZE(objects);
    PyGC_Head **headers = PyMem_RawMalloc(size * sizeof(*headers));
    if (!headers) return PyErr_NoMemory();
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *obj = PyList_GET_ITEM(objects, i);
        if (!PyObject_GC_IsTracked(obj)) {
            PyMem_RawFree(headers);
            PyErr_SetString(PyExc_ValueError, "Snapshot contains an untracked object");
            return NULL;
        }
        headers[i] = _Py_AS_GC(obj);
    }
    double wall[3], cpu[3];
    Py_ssize_t count = 0;
    uintptr_t checksum[2] = {0, 0};
    PyGC_Head *start = headers[0], *node = start;
    double w = now(CLOCK_MONOTONIC), c = now(CLOCK_THREAD_CPUTIME_ID);
    do {
        node = _PyGCHead_NEXT(node);
        count++;
    } while (node != start && count <= size + 16);
    cpu[0] = now(CLOCK_THREAD_CPUTIME_ID) - c;
    wall[0] = now(CLOCK_MONOTONIC) - w;
    if (node != start || count != size + 1) {
        PyMem_RawFree(headers);
        PyErr_Format(PyExc_RuntimeError, "GC list/snapshot mismatch: %zd links, %zd objects", count, size);
        return NULL;
    }
    for (int pass = 0; pass < 2; pass++) {
        if (pass) qsort(headers, size, sizeof(*headers), address_order);
        w = now(CLOCK_MONOTONIC);
        c = now(CLOCK_THREAD_CPUTIME_ID);
        for (Py_ssize_t i = 0; i < size; i++) {
            checksum[pass] ^= headers[i]->_gc_next;
        }
        cpu[pass + 1] = now(CLOCK_THREAD_CPUTIME_ID) - c;
        wall[pass + 1] = now(CLOCK_MONOTONIC) - w;
    }
    PyMem_RawFree(headers);
    if (checksum[0] != checksum[1]) {
        PyErr_SetString(PyExc_RuntimeError, "Header checksum changed during read-only scan");
        return NULL;
    }
    return Py_BuildValue("{s:n,s:n,s:(ddd),s:(ddd)}", "objects", size, "links", count,
                         "wall_seconds", wall[0], wall[1], wall[2],
                         "thread_cpu_seconds", cpu[0], cpu[1], cpu[2]);
}

static PyMethodDef methods[] = {
    {"memory", memory_control, METH_NOARGS, "Time a fixed private 64-MiB pointer graph, independent of the GC heap."},
    {"measure", measure, METH_O, "Read-only list and header timing."},
    {"cold", cold, METH_NOARGS, "Read generation-2 links before any diagnostic object scan."},
    {NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_gc_walk", NULL, -1, methods};
PyMODINIT_FUNC PyInit__gc_walk(void) { return PyModule_Create(&module); }
