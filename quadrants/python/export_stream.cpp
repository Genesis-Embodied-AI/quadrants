/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/python/export.h"
#include "quadrants/program/program.h"

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
}

}  // namespace quadrants
