#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

// My own implementation of LU decomposition with partial pivoting
bool LU(const MatrixXf &a, MatrixXf &l, MatrixXf &u, MatrixXf &p) {
	assert(a.rows() == a.cols());

	// indexes for row exchange
	// NOTE: we will exchange rows of U after decomposition
	ArrayXi index(a.rows());
	for (int r = 0; r < a.rows(); r++) {
		index(r) = r;
	}

	// upper triangular matrix
	u = a;

	// lower triangular matrix
	l = MatrixXf::Identity(a.rows(), a.cols());

	for (int i = 0; i < a.rows(); i++) {
		// find real pivot index
		int pivot_index = index[i];
		float maximum = abs(u(pivot_index, i));
		for (int r = i + 1; r < a.rows(); r++) {
			float value = abs(u(index[r], i));

			if (value > maximum) {
				pivot_index = index[r];
				maximum = value;
			}
		}

		// if row exchange is required
		if (pivot_index > i) {
			// swap pivot index with current row index
			std::swap(index[i], index[pivot_index]);

			// swap row in L
			for (int c = 0; c < i; c++) {
				std::swap(l(i, c), l(pivot_index, c));
			}
		}

		// get pivot value
		float pivot = u(index[i], i);
		if (pivot == 0.0f) {
			// abandon decomposition if pivot does not exist
			return false;
		}

		for (int r = i + 1; r < a.rows(); r++) {
			// scaler for subtract a row
			float scaler = u(index[r], i) / pivot;

			// subtract a row from scaled pivot row
			for (int c = i; c < a.cols(); c++) {
				u(index[r], c) -= scaler * u(index[i], c);
			}

			// scaler matches with L component
			l(r, i) = scaler;
		}
	}

	// permutation matrix P
	p = MatrixXf::Zero(a.rows(), a.cols());
	for (int r = 0; r < a.rows(); r++) {
		p.row(r)[index[r]] = 1.0f;
	}

	// permutate matrix U using P
	u = p * u;

	return true;
}

// Example code for my own LU decomposition
void TestLU() {
	const int DIMENSION = 5;

	MatrixXf a(DIMENSION, DIMENSION);
	a << MatrixXf::Random(DIMENSION, DIMENSION);

	VectorXf b(DIMENSION);
	b << VectorXf::Random(DIMENSION);

	VectorXf c(DIMENSION);
	c << VectorXf::Random(DIMENSION);

	MatrixXf l(DIMENSION, DIMENSION);
	MatrixXf u(DIMENSION, DIMENSION);
	MatrixXf p(DIMENSION, DIMENSION);
	bool invertible = ::LU(a, l, u, p);

	if (!invertible) {
		std::cout << "A is not invertible !!\n";
		return;
	}

	std::cout << "A =\n";
	std::cout << a << "\n\n";

	std::cout << "b =\n";
	std::cout << b << "\n\n";

	std::cout << "A * b =\n";
	std::cout << a * b << "\n\n";

	// PA = LU
	// A = P^T LU
	std::cout << "P^T * L * U * b =\n";
	std::cout << p.transpose() * l * u * b << "\n\n";;
}

int main() {
	srand((unsigned int)time(0));

	TestLU();

	getchar();

	return 0;
}
