#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

// My own implementation of LU decomposition with partial pivoting
bool LU(const MatrixXf &a, MatrixXf &l, MatrixXf &u, MatrixXf &p) {
	// 'a' should be a square matrix, so does 'l', 'u', and 'p'
	assert(a.rows() == a.cols());
	int size = a.rows();

	// no pivoting upper triangular matrix 'nu' elimination begin with 'a'
	MatrixXf nu = a;

	// row_indexes has row order of 'nu'
	ArrayXi row_indexes(size);
	for (int r = 0; r < size; r++) {
		row_indexes(r) = r;
	}

	// lower triangular matrix
	l = MatrixXf::Identity(size, size);

	// 'i' is diagonal index of the matrix 
	for (int i = 0; i < size; i++) {
		// find real pivot index
		int pivot_index = row_indexes[i];
		float maximum = abs(nu(pivot_index, i));
		for (int r = i + 1; r < size; r++) {
			float value = abs(nu(row_indexes[r], i));

			if (value > maximum) {
				pivot_index = row_indexes[r];
				maximum = value;
			}
		}

		// if row exchange is required
		if (pivot_index > i) {
			// swap pivot index with current row index
			std::swap(row_indexes[i], row_indexes[pivot_index]);

			// swap rows in L
			for (int c = 0; c < i; c++) {
				std::swap(l(i, c), l(pivot_index, c));
			}
		}

		// get pivot value
		float pivot = nu(row_indexes[i], i);
		if (pivot == 0.0f) {
			// abandon decomposition if pivot does not exist
			return false;
		}

		for (int r = i + 1; r < size; r++) {
			// scaler for subtract a row
			float scaler = nu(row_indexes[r], i) / pivot;

			// subtract a row from scaled pivot row
			for (int c = i; c < size; c++) {
				nu(row_indexes[r], c) -= scaler * nu(row_indexes[i], c);
			}

			// scaler matches with L component
			l(r, i) = scaler;
		}
	}

	// generate permutation matrix P with 'row_indexes'
	p = MatrixXf::Zero(size, size);
	for (int r = 0; r < size; r++) {
		p.row(r)[row_indexes[r]] = 1.0f;
	}

	// permutate matrix U using P
#if 0
	u = p * nu;
#else
	for (int r = 0; r < size; r++) {
		u.row(r) = nu.row(row_indexes[r]);
	}
#endif

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
	bool invertible = ::LU(a, l, u, p);

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
