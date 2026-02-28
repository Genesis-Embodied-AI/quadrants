#pragma once

#include "sparse_matrix.h"

#include "quadrants/ir/type.h"
#include "quadrants/program/program.h"

#define DECLARE_EIGEN_LLT_SOLVER(dt, type, order)                    \
  typedef EigenSparseSolver<                                         \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>                                       \
      EigenSparseSolver##dt##type##order;

#define DECLARE_EIGEN_LU_SOLVER(dt, type, order)                              \
  typedef EigenSparseSolver<Eigen::Sparse##type<Eigen::SparseMatrix<dt>,      \
                                                Eigen::order##Ordering<int>>, \
                            Eigen::SparseMatrix<dt>>                          \
      EigenSparseSolver##dt##type##order;

namespace quadrants::lang {

class SparseSolver {
 protected:
  int rows_{0};
  int cols_{0};
  DataType dtype_{PrimitiveType::f32};
  bool is_initialized_{false};

 public:
  virtual ~SparseSolver() = default;
  void init_solver(const int rows, const int cols, const DataType dtype) {
    rows_ = rows;
    cols_ = cols;
    dtype_ = dtype;
  }
  virtual bool compute(const SparseMatrix &sm) = 0;
  virtual void analyze_pattern(const SparseMatrix &sm) = 0;
  virtual void factorize(const SparseMatrix &sm) = 0;
  virtual bool info() = 0;
};

template <class EigenSolver, class EigenMatrix>
class EigenSparseSolver : public SparseSolver {
 private:
  EigenSolver solver_;

 public:
  ~EigenSparseSolver() override = default;
  bool compute(const SparseMatrix &sm) override;
  void analyze_pattern(const SparseMatrix &sm) override;
  void factorize(const SparseMatrix &sm) override;
  template <typename T>
  T solve(const T &b);

  template <typename T, typename V>
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                const Ndarray &x);
  bool info() override;
};

DECLARE_EIGEN_LLT_SOLVER(float32, LLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LLT, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LDLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LDLT, COLAMD);
DECLARE_EIGEN_LU_SOLVER(float32, LU, AMD);
DECLARE_EIGEN_LU_SOLVER(float32, LU, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LLT, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LDLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LDLT, COLAMD);
DECLARE_EIGEN_LU_SOLVER(float64, LU, AMD);
DECLARE_EIGEN_LU_SOLVER(float64, LU, COLAMD);

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering);
}  // namespace quadrants::lang
