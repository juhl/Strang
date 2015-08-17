#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

// Gaussian elimination
// [ A | b ] = [ U | c ]
bool GaussElimination(const MatrixXf &a, const VectorXf &b, MatrixXf &u, VectorXf &c) {
	// 'a' should be a square matrix
	assert(a.rows() == a.cols());

	// left hand side matrix A
	u = a;

	// right hand side vector b
	c = b;

	// 'i' is diagonal index of the matrix 
	for (int i = 0; i < a.rows(); i++) {
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
			// swap rows of U
			u.row(i).swap(u.row(pivot_index));
			// swap rows of c
			std::swap(c(i), c(pivot_index));
		}

		for (int r = i + 1; r < a.rows(); r++) {
			// scaler for subtract a row
			float scaler = u(r, i) / pivot;

			// subtract a row from scaled pivot row
			for (int c = i; c < a.cols(); c++) {
				u(r, c) -= scaler * u(i, c);
			}
			c(r) -= scaler * c(i);
		}
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

	x = a.lu().solve(b);
	std::cout << "Ax = b, x =\n";
	std::cout << x << "\n\n";
}

int main() {
	srand((unsigned int)time(0));

	TestGaussElimination();

	getchar();

	return 0;
}
