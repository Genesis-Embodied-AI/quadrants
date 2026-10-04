/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/python/export.h"
#include "quadrants/program/program.h"

#ifdef QD_WITH_CUDA
#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#endif

namespace quadrants {

void export_stream(nb::module_ &m, nb::class_<lang::Program> &program_class) {
  using lang::Program;
  program_class.def("stream_create", [](Program *p) { return p->stream_manager().create_stream(); })
      .def("stream_destroy", [](Program *p, uint64 h) { p->stream_manager().destroy_stream(h); })
      .def("stream_synchronize", [](Program *p, uint64 h) { p->stream_manager().synchronize_stream(h); })
      .def("set_current_cuda_stream", [](Program *p, uint64 h) { p->stream_manager().set_current_stream(h); })
      .def("event_create", [](Program *p) { return p->stream_manager().create_event(); })
      .def("event_destroy", [](Program *p, uint64 h) { p->stream_manager().destroy_event(h); })
      .def("event_record", [](Program *p, uint64 eh, uint64 sh) { p->stream_manager().record_event(eh, sh); })
      .def("event_synchronize", [](Program *p, uint64 h) { p->stream_manager().synchronize_event(h); })
      .def("stream_wait_event",
           [](Program *p, uint64 sh, uint64 eh) { p->stream_manager().stream_wait_event(sh, eh); });

#ifdef QD_WITH_CUDA
  program_class.def("_cuda_event_query", [](Program *p, uint64 h) {
    QD_ASSERT(p->compile_config().arch == Arch::cuda);
    lang::CUDAContext::get_instance().make_current();
    // Query readiness without treating CUDA_ERROR_NOT_READY as a fatal driver error.
    return lang::CUDADriver::get_instance().event_query.call(reinterpret_cast<void *>(h));
  });
#endif
}

}  // namespace quadrants
