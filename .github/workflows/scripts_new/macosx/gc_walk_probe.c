/* Read-only CPython 3.10 GC-list diagnostic. Never writes interpreter object headers or releases the GIL. */
#define Py_BUILD_CORE
#include <Python.h>
#include <internal/pycore_gc.h>
#include <internal/pycore_interp.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <libproc.h>
#include <sys/mman.h>
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

/* Pin the same generation-2 list under one uninterrupted GIL hold, before returning to Python bytecode. */
static PyObject *snapshot(PyObject *self, PyObject *unused) {
    if (PyGC_IsEnabled()) {
        PyErr_SetString(PyExc_RuntimeError, "Atomic snapshot requires automatic GC disabled");
        return NULL;
    }
    PyObject *timings = cold(self, unused);
    if (!timings) return NULL;
    Py_ssize_t count = PyLong_AsSsize_t(PyDict_GetItemString(timings, "objects"));
    if (count < 0) {
        Py_DECREF(timings);
        return NULL;
    }
    double start = now(CLOCK_MONOTONIC);
    PyObject *objects = PyList_New(count);
    if (!objects) {
        Py_DECREF(timings);
        return NULL;
    }
    PyGC_Head *head = &PyThreadState_Get()->interp->gc.generations[2].head;
    PyGC_Head *node = _PyGCHead_NEXT(head);
    for (Py_ssize_t i = 0; i < count; i++) {
        if (node == head) goto mismatch;
        PyObject *obj = (PyObject *)(node + 1);
        Py_INCREF(obj);
        PyList_SET_ITEM(objects, i, obj);
        node = _PyGCHead_NEXT(node);
    }
    if (node != head) goto mismatch;
    return Py_BuildValue("(NNd)", timings, objects, now(CLOCK_MONOTONIC) - start);

mismatch:
    Py_DECREF(objects);
    Py_DECREF(timings);
    PyErr_SetString(PyExc_RuntimeError, "Generation-2 count changed inside atomic snapshot");
    return NULL;
}

/* A fixed private graph independent of Python objects and Vulkan. Retained for this process lifetime. */
typedef struct MemoryNode {
    struct MemoryNode *volatile next;
    unsigned char padding[64 - sizeof(void *)];
} MemoryNode;
_Static_assert(sizeof(MemoryNode) == 64, "Control nodes must have a fixed stride");
static MemoryNode *memory_nodes = NULL, *memory_head = NULL, *memory_expected = NULL;
#define MEMORY_CHUNKS 64
static unsigned memory_chunk_new_pages[MEMORY_CHUNKS];
static long memory_page_size;

/* Guest residency query: does not establish residency or scheduling on the hypervisor host. */
static PyObject *memory_residency(void) {
#ifdef __APPLE__
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return PyErr_SetFromErrno(PyExc_OSError);
    uintptr_t begin = (uintptr_t)memory_nodes & ~((uintptr_t)page_size - 1);
    uintptr_t end = ((uintptr_t)memory_nodes + (1u << 20) * sizeof(MemoryNode) + page_size - 1)
                    & ~((uintptr_t)page_size - 1);
    size_t pages = (end - begin) / page_size;
    char *vector = calloc(pages, 1);
    if (!vector) return PyErr_NoMemory();
    int status = mincore((void *)begin, end - begin, vector), saved_errno = status ? errno : 0;
    size_t resident = 0, paged_out = 0;
    if (!status) {
        for (size_t i = 0; i < pages; i++) {
            resident += !!(vector[i] & MINCORE_INCORE);
            paged_out += !!(vector[i] & MINCORE_PAGED_OUT);
        }
    }
    free(vector);
    return Py_BuildValue("{s:i,s:i,s:l,s:n,s:n,s:n}", "status", status, "errno", saved_errno,
                         "page_size", page_size, "pages", (Py_ssize_t)pages,
                         "resident", (Py_ssize_t)resident, "paged_out", (Py_ssize_t)paged_out);
#else
    Py_RETURN_NONE;
#endif
}

static double usage_seconds(struct timeval value) {
    return value.tv_sec + value.tv_usec * 1e-6;
}

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
#ifdef __APPLE__
        memory_page_size = sysconf(_SC_PAGESIZE);
#else
        memory_page_size = 4096;
#endif
        if (memory_page_size <= 0) return PyErr_SetFromErrno(PyExc_OSError);
        uintptr_t first_page = (uintptr_t)nodes / memory_page_size;
        size_t page_count = ((uintptr_t)(nodes + count) - 1) / memory_page_size - first_page + 1;
        unsigned char *seen = calloc(page_count, 1);
        if (!seen) {
            PyMem_RawFree(order);
            return PyErr_NoMemory();
        }
        for (uint32_t i = 0; i < visits; i++) {
            size_t page = (uintptr_t)&nodes[order[i]] / memory_page_size - first_page;
            if (!seen[page]) {
                seen[page] = 1;
                memory_chunk_new_pages[i / (visits / MEMORY_CHUNKS)]++;
            }
        }
        free(seen);
        PyMem_RawFree(order);
    }
    PyObject *residency_before = memory_residency();
    if (!residency_before) return NULL;
    PyObject *resources = PyList_New(0);
    if (!resources) {
        Py_DECREF(residency_before);
        return NULL;
    }
    double wall[4], cpu[4];
    double chunk_wall[2][MEMORY_CHUNKS], chunk_cpu[2][MEMORY_CHUNKS];
    uint64_t checksums[2];
    for (int pass = 0; pass < 4; pass++) {
        struct rusage before, after;
        if (getrusage(RUSAGE_SELF, &before)) {
            PyErr_SetFromErrno(PyExc_OSError);
            goto failed;
        }
        double w = now(CLOCK_MONOTONIC), c = now(CLOCK_THREAD_CPUTIME_ID);
        if (pass == 0 || pass == 3) {
            checksums[pass / 3] = arithmetic();
        } else {
            MemoryNode *node = memory_head;
            for (unsigned chunk = 0; chunk < MEMORY_CHUNKS; chunk++) {
                double cw = now(CLOCK_MONOTONIC), cc = now(CLOCK_THREAD_CPUTIME_ID);
                for (uint32_t i = 0; i < visits / MEMORY_CHUNKS; i++) node = node->next;
                chunk_cpu[pass - 1][chunk] = now(CLOCK_THREAD_CPUTIME_ID) - cc;
                chunk_wall[pass - 1][chunk] = now(CLOCK_MONOTONIC) - cw;
            }
            if (node != memory_expected) {
                PyErr_SetString(PyExc_RuntimeError, "Fixed memory control checksum mismatch");
                goto failed;
            }
        }
        cpu[pass] = now(CLOCK_THREAD_CPUTIME_ID) - c;
        wall[pass] = now(CLOCK_MONOTONIC) - w;
        if (getrusage(RUSAGE_SELF, &after)) {
            PyErr_SetFromErrno(PyExc_OSError);
            goto failed;
        }
        PyObject *delta = Py_BuildValue("{s:l,s:l,s:l,s:l,s:d,s:d}",
                                       "minor_faults", after.ru_minflt - before.ru_minflt,
                                       "major_faults", after.ru_majflt - before.ru_majflt,
                                       "voluntary_switches", after.ru_nvcsw - before.ru_nvcsw,
                                       "involuntary_switches", after.ru_nivcsw - before.ru_nivcsw,
                                       "user_cpu", usage_seconds(after.ru_utime) - usage_seconds(before.ru_utime),
                                       "system_cpu", usage_seconds(after.ru_stime) - usage_seconds(before.ru_stime));
        if (!delta) goto failed;
        int append_status = PyList_Append(resources, delta);
        Py_DECREF(delta);
        if (append_status) goto failed;
    }
    if (checksums[0] != checksums[1]) {
        PyErr_SetString(PyExc_RuntimeError, "Memory-control arithmetic checksum mismatch");
        goto failed;
    }
    PyObject *residency_after = memory_residency();
    if (!residency_after) goto failed;
    PyObject *chunks = PyList_New(MEMORY_CHUNKS);
    if (!chunks) {
        Py_DECREF(residency_after);
        goto failed;
    }
    for (unsigned chunk = 0; chunk < MEMORY_CHUNKS; chunk++) {
        PyObject *item = Py_BuildValue("{s:i,s:i,s:i,s:(dd),s:(dd)}", "index", (int)chunk,
                                      "visits", (int)(visits / MEMORY_CHUNKS),
                                      "new_pages", (int)memory_chunk_new_pages[chunk],
                                      "wall_seconds", chunk_wall[0][chunk], chunk_wall[1][chunk],
                                      "thread_cpu_seconds", chunk_cpu[0][chunk], chunk_cpu[1][chunk]);
        if (!item) {
            Py_DECREF(chunks);
            Py_DECREF(residency_after);
            goto failed;
        }
        PyList_SET_ITEM(chunks, chunk, item);
    }
    return Py_BuildValue("{s:i,s:i,s:i,s:(dddd),s:(dddd),s:N,s:N,s:N,s:N,s:l}", "nodes", (int)count,
                         "visits", (int)visits, "bytes", (int)(count * sizeof(MemoryNode)),
                         "wall_seconds", wall[0], wall[1], wall[2], wall[3],
                         "thread_cpu_seconds", cpu[0], cpu[1], cpu[2], cpu[3],
                         "resources", resources, "residency_before", residency_before,
                         "residency_after", residency_after, "chunks", chunks, "chunk_page_size", memory_page_size);
failed:
    Py_DECREF(resources);
    Py_DECREF(residency_before);
    return NULL;
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

typedef struct ReferenceCounts {
    Py_ssize_t total;
    Py_ssize_t tracked;
} ReferenceCounts;

static int count_reference(PyObject *obj, void *argument) {
    ReferenceCounts *counts = argument;
    counts->total++;
    counts->tracked += PyObject_GC_IsTracked(obj) != 0;
    return 0;
}

static PyObject *references(PyObject *self, PyObject *objects) {
    if (!PyList_CheckExact(objects)) {
        PyErr_SetString(PyExc_ValueError, "Expected a retained GC snapshot");
        return NULL;
    }
    ReferenceCounts counts = {0, 0};
    Py_ssize_t maximum = 0, size = PyList_GET_SIZE(objects);
    const char *maximum_type = "";
    double w = now(CLOCK_MONOTONIC), c = now(CLOCK_THREAD_CPUTIME_ID);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *obj = PyList_GET_ITEM(objects, i);
        traverseproc traverse = Py_TYPE(obj)->tp_traverse;
        if (!traverse) {
            PyErr_SetString(PyExc_RuntimeError, "Tracked object has no traverse function");
            return NULL;
        }
        Py_ssize_t before = counts.total;
        if (traverse(obj, count_reference, &counts)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Reference-count visitor unexpectedly failed");
            return NULL;
        }
        if (counts.total - before > maximum) {
            maximum = counts.total - before;
            maximum_type = Py_TYPE(obj)->tp_name;
        }
    }
    double cpu = now(CLOCK_THREAD_CPUTIME_ID) - c, wall = now(CLOCK_MONOTONIC) - w;
    return Py_BuildValue("{s:n,s:n,s:n,s:n,s:s,s:d,s:d}", "objects", size, "references", counts.total,
                         "tracked_references", counts.tracked, "max_object_references", maximum,
                         "max_object_type", maximum_type, "wall_seconds", wall, "thread_cpu_seconds", cpu);
}

static PyMethodDef methods[] = {
    {"snapshot", snapshot, METH_NOARGS, "Direct-head measurement and pinned snapshot under one GIL hold."},
    {"references", references, METH_O, "Read-only tp_traverse visit counts for a retained object snapshot."},
    {"memory", memory_control, METH_NOARGS, "Time a fixed private 64-MiB pointer graph, independent of the GC heap."},
    {"measure", measure, METH_O, "Read-only list and header timing."},
    {"cold", cold, METH_NOARGS, "Read generation-2 links before any diagnostic object scan."},
    {NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_gc_walk", NULL, -1, methods};
PyMODINIT_FUNC PyInit__gc_walk(void) { return PyModule_Create(&module); }
