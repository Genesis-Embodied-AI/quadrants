#include "quadrants/ir/type_utils.h"

#include "sparse_solver.h"

#include <unordered_map>

namespace quadrants::lang {
#define EIGEN_LLT_SOLVER_INSTANTIATION(dt, type, order)              \
  template class EigenSparseSolver<                                  \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>;
#define EIGEN_LU_SOLVER_INSTANTIATION(dt, type, order)  \
  template class EigenSparseSolver<                     \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,      \
                          Eigen::order##Ordering<int>>, \
      Eigen::SparseMatrix<dt>>;
// Explicit instantiation of EigenSparseSolver
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LLT, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LDLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LDLT, COLAMD);
EIGEN_LU_SOLVER_INSTANTIATION(float32, LU, AMD);
EIGEN_LU_SOLVER_INSTANTIATION(float32, LU, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LLT, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LDLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LDLT, COLAMD);
EIGEN_LU_SOLVER_INSTANTIATION(float64, LU, AMD);
EIGEN_LU_SOLVER_INSTANTIATION(float64, LU, COLAMD);
}  // namespace quadrants::lang

// Explicit instantiation of the template class EigenSparseSolver::solve
#define EIGEN_LLT_SOLVE_INSTANTIATION(dt, type, order, df)               \
  using T##dt = Eigen::VectorX##df;                                      \
  using S##dt##type##order =                                             \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower,     \
                              Eigen::order##Ordering<int>>;              \
  template T##dt                                                         \
  EigenSparseSolver<S##dt##type##order, Eigen::SparseMatrix<dt>>::solve( \
      const T##dt &b);
#define EIGEN_LU_SOLVE_INSTANTIATION(dt, type, order, df)                  \
  using LUT##dt = Eigen::VectorX##df;                                      \
  using LUS##dt##type##order =                                             \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,                         \
                          Eigen::order##Ordering<int>>;                    \
  template LUT##dt                                                         \
  EigenSparseSolver<LUS##dt##type##order, Eigen::SparseMatrix<dt>>::solve( \
      const LUT##dt &b);

// Explicit instantiation of the template class EigenSparseSolver::solve_rf
#define INSTANTIATE_LLT_SOLVE_RF(dt, type, order, df)                     \
  using llt##dt##type##order =                                            \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower,      \
                              Eigen::order##Ordering<int>>;               \
  template void EigenSparseSolver<llt##dt##type##order,                   \
                                  Eigen::SparseMatrix<dt>>::solve_rf<df,  \
                                                                     dt>( \
      Program * prog, const SparseMatrix &sm, const Ndarray &b,           \
      const Ndarray &x);

#define INSTANTIATE_LU_SOLVE_RF(dt, type, order, df)                      \
  using lu##dt##type##order =                                             \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,                        \
                          Eigen::order##Ordering<int>>;                   \
  template void EigenSparseSolver<lu##dt##type##order,                    \
                                  Eigen::SparseMatrix<dt>>::solve_rf<df,  \
                                                                     dt>( \
      Program * prog, const SparseMatrix &sm, const Ndarray &b,           \
      const Ndarray &x);

#define MAKE_EIGEN_SOLVER(dt, type, order) \
  std::make_unique<EigenSparseSolver##dt##type##order>()

#define MAKE_SOLVER(dt, type, order)                             \
  {{#dt, #type, #order}, []() -> std::unique_ptr<SparseSolver> { \
     return MAKE_EIGEN_SOLVER(dt, type, order);                  \
   }}

using Triplets = std::tuple<std::string, std::string, std::string>;
namespace {
struct key_hash {
  std::size_t operator()(const Triplets &k) const {
    auto h1 = std::hash<std::string>{}(std::get<0>(k));
    auto h2 = std::hash<std::string>{}(std::get<1>(k));
    auto h3 = std::hash<std::string>{}(std::get<2>(k));
    return h1 ^ h2 ^ h3;
  }
};
}  // namespace

namespace quadrants::lang {

#define GET_EM(sm) \
  const EigenMatrix *mat = (const EigenMatrix *)(sm.get_matrix());

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::compute(
    const SparseMatrix &sm) {
  if (!is_initialized_) {
    SparseSolver::init_solver(sm.num_rows(), sm.num_cols(), sm.get_data_type());
  }
  GET_EM(sm);
  solver_.compute(*mat);
  if (solver_.info() != Eigen::Success) {
    return false;
  } else
    return true;
}
template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::analyze_pattern(
    const SparseMatrix &sm) {
  if (!is_initialized_) {
    SparseSolver::init_solver(sm.num_rows(), sm.num_cols(), sm.get_data_type());
  }
  GET_EM(sm);
  solver_.analyzePattern(*mat);
}

template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::factorize(
    const SparseMatrix &sm) {
  GET_EM(sm);
  solver_.factorize(*mat);
}

template <class EigenSolver, class EigenMatrix>
template <typename T>
T EigenSparseSolver<EigenSolver, EigenMatrix>::solve(const T &b) {
  return solver_.solve(b);
}

EIGEN_LLT_SOLVE_INSTANTIATION(float32, LLT, AMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LLT, COLAMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LDLT, AMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LDLT, COLAMD, f);
EIGEN_LU_SOLVE_INSTANTIATION(float32, LU, AMD, f);
EIGEN_LU_SOLVE_INSTANTIATION(float32, LU, COLAMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LLT, AMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LLT, COLAMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LDLT, AMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LDLT, COLAMD, d);
EIGEN_LU_SOLVE_INSTANTIATION(float64, LU, AMD, d);
EIGEN_LU_SOLVE_INSTANTIATION(float64, LU, COLAMD, d);

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::info() {
  return solver_.info() == Eigen::Success;
}

template <class EigenSolver, class EigenMatrix>
template <typename T, typename V>
void EigenSparseSolver<EigenSolver, EigenMatrix>::solve_rf(
    Program *prog,
    const SparseMatrix &sm,
    const Ndarray &b,
    const Ndarray &x) {
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  Eigen::Map<T>((V *)dX, rows_) = solver_.solve(Eigen::Map<T>((V *)db, cols_));
}

INSTANTIATE_LLT_SOLVE_RF(float32, LLT, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LDLT, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LLT, AMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LDLT, AMD, Eigen::VectorXf)
INSTANTIATE_LU_SOLVE_RF(float32, LU, AMD, Eigen::VectorXf)
INSTANTIATE_LU_SOLVE_RF(float32, LU, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float64, LLT, COLAMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LDLT, COLAMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LLT, AMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LDLT, AMD, Eigen::VectorXd)
INSTANTIATE_LU_SOLVE_RF(float64, LU, AMD, Eigen::VectorXd)
INSTANTIATE_LU_SOLVE_RF(float64, LU, COLAMD, Eigen::VectorXd)

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering) {
  using key_type = Triplets;
  using func_type = std::unique_ptr<SparseSolver> (*)();
  static const std::unordered_map<key_type, func_type, key_hash>
      solver_factory = {
          MAKE_SOLVER(float32, LLT, AMD),  MAKE_SOLVER(float32, LLT, COLAMD),
          MAKE_SOLVER(float32, LDLT, AMD), MAKE_SOLVER(float32, LDLT, COLAMD),
          MAKE_SOLVER(float64, LLT, AMD),  MAKE_SOLVER(float64, LLT, COLAMD),
          MAKE_SOLVER(float64, LDLT, AMD), MAKE_SOLVER(float64, LDLT, COLAMD)};
  static const std::unordered_map<std::string, std::string> dt_map = {
      {"f32", "float32"}, {"f64", "float64"}};
  auto it = dt_map.find(quadrants::lang::data_type_name(dt));
  if (it == dt_map.end())
    QD_ERROR("Not supported sparse solver data type: {}",
             quadrants::lang::data_type_name(dt));

  Triplets solver_key = std::make_tuple(it->second, solver_type, ordering);
  if (solver_factory.find(solver_key) != solver_factory.end()) {
    auto solver_func = solver_factory.at(solver_key);
    return solver_func();
  } else if (solver_type == "LU") {
    if (it->first == "f32") {
      using EigenMatrix = Eigen::SparseMatrix<float32>;
      using LU = Eigen::SparseLU<EigenMatrix>;
      return std::make_unique<EigenSparseSolver<LU, EigenMatrix>>();
    } else if (it->first == "f64") {
      using EigenMatrix = Eigen::SparseMatrix<float64>;
      using LU = Eigen::SparseLU<EigenMatrix>;
      return std::make_unique<EigenSparseSolver<LU, EigenMatrix>>();
    } else {
      QD_ERROR("Not supported sparse solver data type: {}", it->second);
    }
  } else
    QD_ERROR("Not supported sparse solver type: {}", solver_type);
}

}  // namespace quadrants::lang
