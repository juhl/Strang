#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

// Gaussian elimination
bool GaussElimination(const MatrixXf &a, const VectorXf &b, MatrixXf &u, VectorXf &c) {
	// 'a' should be a square matrix, so does 'l', 'u', and 'p'
	assert(a.rows() == a.cols());

	c = b;

	// no pivoting upper triangular matrix 'nu' elimination begin with 'a'
	MatrixXf nu = a;

	// row_indexes has row order of 'nu'
	ArrayXi row_indexes(a.rows());
	for (int r = 0; r < a.rows(); r++) {
		row_indexes(r) = r;
	}

	// 'i' is diagonal index of the matrix 
	for (int i = 0; i < a.rows(); i++) {
		// find real pivot index
		int pivot_index = row_indexes[i];
		float maximum = abs(nu(pivot_index, i));
		for (int r = i + 1; r < a.rows(); r++) {
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
			std::swap(c[i], c[pivot_index]);
		}

		// get pivot value
		float pivot = nu(row_indexes[i], i);
		if (pivot == 0.0f) {
			// abandon decomposition if pivot does not exist
			return false;
		}

		for (int r = i + 1; r < a.rows(); r++) {
			// scaler for subtract a row
			float scaler = nu(row_indexes[r], i) / pivot;

			// subtract a row from scaled pivot row
			for (int c = i; c < a.cols(); c++) {
				nu(row_indexes[r], c) -= scaler * nu(row_indexes[i], c);
			}
			c[r] -= scaler * c[i];
		}
	}

	for (int r = 0; r < a.rows(); r++) {
		u.row(r) = nu.row(row_indexes[r]);
	}

	return true;
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

// Solve Ax = b with gaussian elimination
bool SolveGaussElimination(const MatrixXf &a, const VectorXf &b, VectorXf &x) {
	MatrixXf u(a.rows(), a.cols());
	VectorXf c(b.rows());

	if (!GaussElimination(a, b, u, c)) {
		return false;
	}

	x = SolveUTriangular(u, c);

	return true;
}

void TestGaussElimination() {
	const int DIMENSION = 4;

	MatrixXf a(DIMENSION, DIMENSION);
	a << MatrixXf::Random(DIMENSION, DIMENSION);

	VectorXf b(DIMENSION);
	b << VectorXf::Random(DIMENSION);

	VectorXf x;
	bool invertible = ::SolveGaussElimination(a, b, x);

	if (!invertible) {
		std::cout << "ERROR: A is not invertible !!\n";
		return;
	}

	std::cout << "A =\n";
	std::cout << a << "\n\n";

	std::cout << "b =\n";
	std::cout << b << "\n\n";

	std::cout << "Ax = b, x =\n";
	std::cout << x << "\n\n";

	PartialPivLU<MatrixXf> decomp(a);
	x = decomp.solve(b);
	std::cout << "Ax = b, x =\n";
	std::cout << x << "\n\n";
}

int main() {
	srand((unsigned int)time(0));

	TestGaussElimination();

	getchar();

	return 0;
}