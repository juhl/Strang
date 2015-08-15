#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

// My own implementation of LU decomposition with partial pivoting
bool LU(const MatrixXf &a, MatrixXf &l, MatrixXf &u, MatrixXf &p) {
	assert(a.rows() == a.cols());
	int size = a.rows();

	// indexes for row exchange
	// NOTE: we will exchange rows of U after decomposition
	ArrayXi indexes(size);
	for (int r = 0; r < size; r++) {
		indexes(r) = r;
	}

	// un-permutated upper triangular matrix
	// 'indexes' has row order 
	MatrixXf uu = a;

	// lower triangular matrix
	l = MatrixXf::Identity(size, size);

	// 'i' is diagonal index of the matrix 
	for (int i = 0; i < size; i++) {
		// find real pivot index
		int pivot_index = indexes[i];
		float maximum = abs(uu(pivot_index, i));
		for (int r = i + 1; r < size; r++) {
			float value = abs(uu(indexes[r], i));

			if (value > maximum) {
				pivot_index = indexes[r];
				maximum = value;
			}
		}

		// if row exchange is required
		if (pivot_index > i) {
			// swap pivot index with current row index
			std::swap(indexes[i], indexes[pivot_index]);

			// swap rows in L
			for (int c = 0; c < i; c++) {
				std::swap(l(i, c), l(pivot_index, c));
			}
		}

		// get pivot value
		float pivot = uu(indexes[i], i);
		if (pivot == 0.0f) {
			// abandon decomposition if pivot does not exist
			return false;
		}

		for (int r = i + 1; r < size; r++) {
			// scaler for subtract a row
			float scaler = uu(indexes[r], i) / pivot;

			// subtract a row from scaled pivot row
			for (int c = i; c < size; c++) {
				uu(indexes[r], c) -= scaler * uu(indexes[i], c);
			}

			// scaler matches with L component
			l(r, i) = scaler;
		}
	}

	// generate permutation matrix P with 'indexes'
	p = MatrixXf::Zero(size, size);
	for (int r = 0; r < size; r++) {
		p.row(r)[indexes[r]] = 1.0f;
	}

	// permutate matrix U using P
#if 0
	u = p * u;
#else
	for (int r = 0; r < size; r++) {
		u.row(r) = uu.row(indexes[r]);
	}
#endif

	return true;
}

// Example code for my own LU decomposition
void TestLU() {
	const int DIMENSION = 5;

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
