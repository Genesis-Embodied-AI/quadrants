/* Read-only CPython 3.10 GC-list diagnostic. Never writes interpreter object headers or releases the GIL. */
#define Py_BUILD_CORE
#include <Python.h>
#include <internal/pycore_gc.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

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

static PyMethodDef methods[] = {{"measure", measure, METH_O, "Read-only list and header timing."}, {NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_gc_walk", NULL, -1, methods};
PyMODINIT_FUNC PyInit__gc_walk(void) { return PyModule_Create(&module); }
