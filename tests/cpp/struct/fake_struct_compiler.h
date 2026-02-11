#include "quadrants/struct/struct.h"

namespace quadrants::lang {

class FakeStructCompiler : public StructCompiler {
 public:
  void generate_types(SNode &) override {
  }

  void generate_child_accessors(SNode &) override {
  }

  void run(SNode &root) override {
  }
};

}  // namespace quadrants::lang
