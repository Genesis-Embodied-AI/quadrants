#ifdef QD_WITH_LLVM
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#endif

#include "quadrants/ir/transforms.h"
#include "quadrants/util/file_sequence_writer.h"

namespace quadrants {

FileSequenceWriter::FileSequenceWriter(std::string filename_template,
                                       std::string file_type)
    : counter_(0),
      filename_template_(filename_template),
      file_type_(file_type) {
}

#ifdef QD_WITH_LLVM
std::string FileSequenceWriter::write(llvm::Module *module) {
  std::string str;
  llvm::raw_string_ostream ros(str);
  module->print(ros, nullptr);
  return write(str);
}
#endif

std::string FileSequenceWriter::write(const std::string &str) {
  auto [ofs, fn] = create_new_file();
  ofs << str;
  return fn;
}

std::string FileSequenceWriter::write(lang::IRNode *irnode) {
  std::string content;
  lang::irpass::print(irnode, &content);
  return write(content);
}

std::pair<std::ofstream, std::string> FileSequenceWriter::create_new_file() {
  auto fn = fmt::format(filename_template_, counter_);
  QD_INFO("Saving {} to {}", file_type_, fn);
  counter_++;
  return {std::ofstream(fn), fn};
}

}  // namespace quadrants
