#include "quadrants/program/sparse_matrix.h"

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "Eigen/Dense"
#include "Eigen/SparseLU"
#include "quadrants/rhi/cuda/cuda_driver.h"

#define BUILD(TYPE)                                                         \
  {                                                                         \
    using T = Eigen::Triplet<float##TYPE>;                                  \
    std::vector<T> *triplets = static_cast<std::vector<T> *>(triplets_adr); \
    matrix_.setFromTriplets(triplets->begin(), triplets->end());            \
  }

#define MAKE_MATRIX(TYPE, STORAGE)                                        \
  {Pair("f" #TYPE, #STORAGE),                                             \
   [](int rows, int cols, DataType dt) -> std::unique_ptr<SparseMatrix> { \
     using FC = Eigen::SparseMatrix<float##TYPE, Eigen::STORAGE>;         \
     return std::make_unique<EigenSparseMatrix<FC>>(rows, cols, dt);      \
   }}

#define INSTANTIATE_SPMV(type, storage)                               \
  template void                                                       \
  EigenSparseMatrix<Eigen::SparseMatrix<type, Eigen::storage>>::spmv( \
      Program *prog, const Ndarray &x, const Ndarray &y);

namespace {
using Pair = std::pair<std::string, std::string>;
struct key_hash {
  std::size_t operator()(const Pair &k) const {
    auto h1 = std::hash<std::string>{}(k.first);
    auto h2 = std::hash<std::string>{}(k.second);
    return h1 ^ h2;
  }
};

template <typename T, typename T1, typename T2>
void print_triplets_from_csr(int64_t n_rows,
                             int n_cols,
                             T *row,
                             T1 *col,
                             T2 *value,
                             std::ostringstream &ostr) {
  using Triplets = Eigen::Triplet<T2>;
  std::vector<Triplets> trips;
  for (int64_t i = 1; i <= n_rows; ++i) {
    auto n_i = row[i] - row[i - 1];
    for (auto j = 0; j < n_i; ++j) {
      trips.push_back({static_cast<int>(i - 1),
                       static_cast<int>(col[row[i - 1] + j]),
                       static_cast<float>(value[row[i - 1] + j])});
    }
  }
  Eigen::SparseMatrix<float> m(n_rows, n_cols);
  m.setFromTriplets(trips.begin(), trips.end());
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  ostr << Eigen::MatrixXf(m.cast<float>()).format(clean_fmt);
}

template <typename T, typename T1, typename T2>
T2 get_element_from_csr(int row,
                        int col,
                        T *row_data,
                        T1 *col_data,
                        T2 *value) {
  for (T i = row_data[row]; i < row_data[row + 1]; ++i) {
    if (col == col_data[i])
      return value[i];
  }
  // zero entry
  return 0;
}

}  // namespace

namespace quadrants::lang {

SparseMatrixBuilder::SparseMatrixBuilder(int rows,
                                         int cols,
                                         int max_num_triplets,
                                         DataType dtype,
                                         const std::string &storage_format)
    : rows_(rows),
      cols_(cols),
      max_num_triplets_(max_num_triplets),
      dtype_(dtype),
      storage_format_(storage_format) {
  auto element_size = data_type_size(dtype);
  QD_ASSERT((element_size == 4 || element_size == 8));
}

SparseMatrixBuilder::~SparseMatrixBuilder() = default;

void SparseMatrixBuilder::create_ndarray(Program *prog) {
  ndarray_data_base_ptr_ = prog->create_ndarray(
      dtype_, std::vector<int>{3 * (int)max_num_triplets_ + 1});
  ndarray_data_ptr_ = prog->get_ndarray_data_ptr_as_int(ndarray_data_base_ptr_);
}

void SparseMatrixBuilder::delete_ndarray(Program *prog) {
  prog->delete_ndarray(ndarray_data_base_ptr_);
}

template <typename T, typename G>
void SparseMatrixBuilder::print_triplets_template() {
  auto ptr = get_ndarray_data_ptr();
  G *data = reinterpret_cast<G *>(ptr);
  num_triplets_ = data[0];
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  data += 1;
  for (int i = 0; i < num_triplets_; i++) {
    fmt::print("[{}, {}] = {}\n", data[i * 3], data[i * 3 + 1],
               quadrants_union_cast<T>(data[i * 3 + 2]));
  }
}

void SparseMatrixBuilder::print_triplets_eigen() {
  auto element_size = data_type_size(dtype_);
  switch (element_size) {
    case 4:
      print_triplets_template<float32, int32>();
      break;
    case 8:
      print_triplets_template<float64, int64>();
      break;
    default:
      QD_ERROR("Unsupported sparse matrix data type!");
      break;
  }
}

void SparseMatrixBuilder::print_triplets_cuda() {
#ifdef QD_WITH_CUDA
  CUDADriver::get_instance().memcpy_device_to_host(
      &num_triplets_, (void *)get_ndarray_data_ptr(), sizeof(int));
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  auto len = 3 * num_triplets_ + 1;
  std::vector<float32> trips(len);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)trips.data(), (void *)get_ndarray_data_ptr(),
      len * sizeof(float32));
  for (auto i = 0; i < num_triplets_; i++) {
    int row = quadrants_union_cast<int>(trips[3 * i + 1]);
    int col = quadrants_union_cast<int>(trips[3 * i + 2]);
    auto val = trips[i * 3 + 3];
    fmt::print("[{}, {}] = {}\n", row, col, val);
  }
#endif
}

intptr_t SparseMatrixBuilder::get_ndarray_data_ptr() const {
  return ndarray_data_ptr_;
}

template <typename T, typename G>
void SparseMatrixBuilder::build_template(std::unique_ptr<SparseMatrix> &m) {
  using V = Eigen::Triplet<T>;
  std::vector<V> triplets;
  auto ptr = get_ndarray_data_ptr();
  G *data = reinterpret_cast<G *>(ptr);
  num_triplets_ = data[0];
  data += 1;
  for (int i = 0; i < num_triplets_; i++) {
    triplets.push_back(V(data[i * 3], data[i * 3 + 1],
                         quadrants_union_cast<T>(data[i * 3 + 2])));
  }
  m->build_triplets(static_cast<void *>(&triplets));
  clear();
}

std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build() {
  QD_ASSERT(built_ == false);
  built_ = true;
  auto sm = make_sparse_matrix(rows_, cols_, dtype_, storage_format_);
  auto element_size = data_type_size(dtype_);
  switch (element_size) {
    case 4:
      build_template<float32, int32>(sm);
      break;
    case 8:
      build_template<float64, int64>(sm);
      break;
    default:
      QD_ERROR("Unsupported sparse matrix data type!");
      break;
  }
  return sm;
}

std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build_cuda() {
  QD_NOT_IMPLEMENTED;
}

void SparseMatrixBuilder::clear() {
  built_ = false;
  ndarray_data_base_ptr_->write_int(std::vector<int>{0}, 0);
  num_triplets_ = 0;
}

template <class EigenMatrix>
const std::string EigenSparseMatrix<EigenMatrix>::to_string() const {
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::ostringstream ostr;
  ostr << Eigen::MatrixXf(matrix_.template cast<float>()).format(clean_fmt);
  return ostr.str();
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::mmwrite(const std::string &filename) {
  std::ofstream file(filename);
  file << "%%MatrixMarket matrix coordinate real general\n%" << std::endl;
  file << matrix_.rows() << " " << matrix_.cols() << " " << matrix_.nonZeros()
       << std::endl;
  for (int k = 0; k < matrix_.outerSize(); ++k) {
    for (typename EigenMatrix::InnerIterator it(matrix_, k); it; ++it) {
      file << it.row() + 1 << " " << it.col() + 1 << " " << it.value()
           << std::endl;
    }
  }
  file.close();
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::build_triplets(void *triplets_adr) {
  std::string sdtype = quadrants::lang::data_type_name(dtype_);
  if (sdtype == "f32") {
    BUILD(32)
  } else if (sdtype == "f64") {
    BUILD(64)
  } else {
    QD_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::spmv(Program *prog,
                                          const Ndarray &x,
                                          const Ndarray &y) {
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t dY = prog->get_ndarray_data_ptr_as_int(&y);
  std::string sdtype = quadrants::lang::data_type_name(dtype_);
  if (sdtype == "f32") {
    Eigen::Map<Eigen::VectorXf>((float *)dY, rows_) =
        matrix_.template cast<float>() *
        Eigen::Map<Eigen::VectorXf>((float *)dX, cols_);
  } else if (sdtype == "f64") {
    Eigen::Map<Eigen::VectorXd>((double *)dY, rows_) =
        matrix_.template cast<double>() *
        Eigen::Map<Eigen::VectorXd>((double *)dX, cols_);
  } else {
    QD_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

INSTANTIATE_SPMV(float32, ColMajor)
INSTANTIATE_SPMV(float32, RowMajor)
INSTANTIATE_SPMV(float64, ColMajor)
INSTANTIATE_SPMV(float64, RowMajor)

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format = "col_major") {
  using func_type = std::unique_ptr<SparseMatrix> (*)(int, int, DataType);
  static const std::unordered_map<Pair, func_type, key_hash> map = {
      MAKE_MATRIX(32, ColMajor), MAKE_MATRIX(32, RowMajor),
      MAKE_MATRIX(64, ColMajor), MAKE_MATRIX(64, RowMajor)};
  std::unordered_map<std::string, std::string> format_map = {
      {"col_major", "ColMajor"}, {"row_major", "RowMajor"}};
  std::string tdt = quadrants::lang::data_type_name(dt);
  Pair key = std::make_pair(tdt, format_map.at(storage_format));
  auto it = map.find(key);
  if (it != map.end()) {
    auto func = map.at(key);
    return func(rows, cols, dt);
  } else
    QD_ERROR("Unsupported sparse matrix data type: {}, storage format: {}", tdt,
             storage_format);
}

template <typename T>
void build_ndarray_template(SparseMatrix &sm,
                            intptr_t data_ptr,
                            size_t num_triplets) {
  using V = Eigen::Triplet<T>;
  std::vector<V> triplets;
  T *data = reinterpret_cast<T *>(data_ptr);
  for (int i = 0; i < num_triplets; i++) {
    triplets.push_back(V(data[i * 3], data[i * 3 + 1],
                         quadrants_union_cast<T>(data[i * 3 + 2])));
  }
  sm.build_triplets(static_cast<void *>(&triplets));
}

void make_sparse_matrix_from_ndarray(Program *prog,
                                     SparseMatrix &sm,
                                     const Ndarray &ndarray) {
  std::string sdtype = quadrants::lang::data_type_name(sm.get_data_type());
  auto data_ptr = prog->get_ndarray_data_ptr_as_int(&ndarray);
  auto num_triplets = ndarray.get_nelement() * ndarray.get_element_size() / 3;
  if (sdtype == "f32") {
    build_ndarray_template<float32>(sm, data_ptr, num_triplets);
  } else if (sdtype == "f64") {
    build_ndarray_template<float64>(sm, data_ptr, num_triplets);
  } else {
    QD_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

}  // namespace quadrants::lang
