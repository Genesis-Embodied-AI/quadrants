import quadrants as qd

qd.init(arch=qd.x64)

n = 8

K = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = qd.field(qd.f32, shape=n)


@qd.kernel
def fill(A: qd.types.sparse_matrix_builder(), b: qd.template(), interval: qd.i32):
    for i in range(n):
        A[i, i] += 2.0

        if i % interval == 0:
            b[i] += 1.0


fill(K, b, 3)

A = K.build()
print("A:")
print(A)
print("b:")
print(b.to_numpy())

print("Sparse matrix-vector multiplication (SpMV): A @ b =")
x = A @ b
print(x)

print("Solving sparse linear systems Ax = b with the solution x:")
solver = qd.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A)
solver.factorize(A)
x = solver.solve(b)
print(x)
isSuccess = solver.info()
print(f"Computation was successful?: {isSuccess}")
