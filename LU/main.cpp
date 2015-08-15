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
