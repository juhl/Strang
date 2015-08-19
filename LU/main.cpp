#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

// Compute LU decomposition with no pivoting on the given square matrix. 
// A = LU
bool LU(const MatrixXf &a, MatrixXf &l, MatrixXf &u) {
	// 'a' should be a square matrix
	assert(a.rows() == a.cols());
	int size = a.rows();

	// upper triangular matrix U
	// A will be converted to the U
	u = a;

	// lower triangular matrix L
	l = MatrixXf::Identity(size, size);

	// 'i' is diagonal index of the matrix 
	for (int i = 0; i < size; i++) {
		// get the pivot
		float pivot = u(i, i);
		if (pivot == 0.0f) {
			// abandon elimination if pivot is not exist
			return false;
		}

		float invPivot = 1.0f / pivot;

		// elimination process
		for (int r = i + 1; r < size; r++) {
			// scalar for subtract a row
			float scalar = u(r, i) * invPivot;

			// subtract first element of the pivot row is zero
			u(r, i) = 0.0f;

			// subtract a row from scaled pivot row
			for (int c = i + 1; c < size; c++) {
				u(r, c) -= scalar * u(i, c);
			}

			// scalar matches with L component
			l(r, i) = scalar;
		}
	}

	return true;
}

// Compute LU decomposition with partial pivoting on the given square matrix.
// PA = LU
bool PartialPivotLU(const MatrixXf &a, MatrixXf &l, MatrixXf &u, MatrixXf &p) {
	// 'a' should be a square matrix
	assert(a.rows() == a.cols());
	int size = a.rows();

	// upper triangular matrix U
	// A will be converted to the U
	u = a;

	// lower triangular matrix L
	l = MatrixXf::Identity(size, size);

	// indexes for row exchanges
	ArrayXi row_indexes(size);
	for (int r = 0; r < size; r++) {
		row_indexes(r) = r;
	}

	// 'i' is diagonal index of the matrix 
	for (int i = 0; i < size; i++) {
		// find pivot index
		int pivot_index = -1;
		float maximum = 0.0f;
		for (int r = i; r < a.rows(); r++) {
			float value = abs(u(r, i));

			if (value > maximum) {
				pivot_index = r;
				maximum = value;
			}
		}

		// get the pivot
		float pivot = u(pivot_index, i);
		if (pivot == 0.0f) {
			// abandon elimination if pivot is not exist
			return false;
		}

		// if row exchange is required
		if (pivot_index > i) {
			// swap row indexes
			std::swap(row_indexes[i], row_indexes[pivot_index]);

			// swap rows of U
			u.row(i).swap(u.row(pivot_index));

			// swap rows of L
			for (int c = 0; c < i; c++) {
				std::swap(l(i, c), l(pivot_index, c));
			}
		}

		float invPivot = 1.0f / pivot;

		// elimination process
		for (int r = i + 1; r < size; r++) {
			// scalar for subtract a row
			float scalar = u(r, i) * invPivot;

			// subtract first element of the pivot row is zero
			u(r, i) = 0.0f;

			// subtract a row from scaled pivot row
			for (int c = i + 1; c < size; c++) {
				u(r, c) -= scalar * u(i, c);
			}

			// scalar matches with L component
			l(r, i) = scalar;
		}
	}

	// generate permutation matrix P with 'row_indexes'
	p = MatrixXf::Zero(size, size);
	for (int r = 0; r < size; r++) {
		p(r, row_indexes[r]) = 1.0f;
	}

	return true;
}

// Solve Lx = b, L is lower triangular matrix
const VectorXf SolveLTriangular(const MatrixXf &l, const VectorXf &b) {
	VectorXf x(b.rows());

	for (int r = 0; r < l.rows(); r++) {
		float k = 0;

		for (int c = 0; c < r; c++) {
			k += l(r, c) * x(c);
		}

		x(r) = (b(r) - k) / l(r, r);
	}

	return x;
}

// Solve Ux = b, U is upper triangular matrix
const VectorXf SolveUTriangular(const MatrixXf &u, const VectorXf &b) {
	VectorXf x(b.rows());

	for (int r = u.rows() - 1; r >= 0; r--) {
		float k = 0;

		for (int c = r + 1; c < u.cols(); c++) {
			k += u(r, c) * x(c);
		}

		x(r) = (b(r) - k) / u(r, r);
	}

	return x;
}

// Solve (P^T)LUx = b
const VectorXf SolvePLU(const MatrixXf &l, const MatrixXf &u, const MatrixXf &p, const VectorXf &b) {
	// Step 1 : LUx = Pb
	// Step 2 : Lc = Pb
	// Step 3 : Ux = c

	// Step 1
	VectorXf pb = p * b;

	// Step 2
	VectorXf c = SolveLTriangular(l, pb);

	// Step 3
	VectorXf x = SolveUTriangular(u, c);

	return x;
}

void TestLU() {
	const int DIMENSION = 4;

	MatrixXf a(DIMENSION, DIMENSION);
	a << MatrixXf::Random(DIMENSION, DIMENSION);

	VectorXf b(DIMENSION);
	b << VectorXf::Random(DIMENSION);

	MatrixXf l(DIMENSION, DIMENSION);
	MatrixXf u(DIMENSION, DIMENSION);
	MatrixXf p(DIMENSION, DIMENSION);

	bool invertible = ::PartialPivotLU(a, l, u, p);
	if (!invertible) {
		std::cout << "ERROR: A is not invertible !!\n";
		return;
	}

	std::cout << "A =\n";
	std::cout << a << "\n\n";

	std::cout << "b =\n";
	std::cout << b << "\n\n";

	std::cout << "A * b =\n";
	std::cout << a * b << "\n\n";

	// PA = LU
	// A = (P^T)LU
	std::cout << "P^T * L * U * b =\n";
	std::cout << p.transpose() * l * u * b << "\n\n";;

	// Solve Ax = b
	VectorXf x = SolvePLU(l, u, p, b);
	std::cout << "Ax = b, x =\n";
	std::cout << x << "\n\n";

	// Solve Ax = b with Eigen solver
	PartialPivLU<MatrixXf> decomp(a);
	x = decomp.solve(b);
	std::cout << "Ax = b, x =\n";
	std::cout << x << "\n\n";
}

int main() {
	srand((unsigned int)time(0));

	TestLU();

	getchar();

	return 0;
}
