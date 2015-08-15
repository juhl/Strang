#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

// Example code for using Eigen
void TestEigen() {
	// fixed size matrix type
	printf("Is Matrix<float, 2, 2> and Matrix2f are same type ? %s\n", typeid(Matrix<float, 2, 2>) == typeid(Matrix2f) ? "yes" : "no");
	printf("Is Matrix<float, 3, 3> and Matrix3f are same type ? %s\n", typeid(Matrix<float, 3, 3>) == typeid(Matrix3f) ? "yes" : "no");
	printf("Is Matrix<float, 4, 4> and Matrix4f are same type ? %s\n", typeid(Matrix<float, 4, 4>) == typeid(Matrix4f) ? "yes" : "no");
	printf("Is Matrix<float, 3, 1> and Vector3f are same type ? %s\n", typeid(Matrix<float, 3, 1>) == typeid(Vector3f) ? "yes" : "no");
	printf("Is Matrix<float, 1, 3> and RowVector3f are same type ? %s\n", typeid(Matrix<float, 1, 3>) == typeid(RowVector3f) ? "yes" : "no");

	// dynamic size matrix type
	printf("Is Matrix<float, Dynamic, Dynamic> and MatrixXf are same type ? %s\n", typeid(Matrix<float, Dynamic, Dynamic>) == typeid(MatrixXf) ? "yes" : "no");
	printf("Is Matrix<float, Dynamic, 1> and VectorXf are same type ? %s\n", typeid(Matrix<float, Dynamic, 1>) == typeid(VectorXf) ? "yes" : "no");
	printf("Is Matrix<float, 1, Dynamic> and RowVectorXf are same type ? %s\n", typeid(Matrix<float, 1, Dynamic>) == typeid(RowVectorXf) ? "yes" : "no");

	// coefficient accessors (matrix component value is called 'coefficient' in Eigen)
	Matrix3f m1;
	m1(0, 0) = 1.0f; m1(0, 1) = 4.0f; m1(0, 2) = 7.0f;
	m1(1, 0) = 2.0f; m1(1, 1) = 5.0f; m1(1, 2) = 8.0f;
	m1(2, 0) = 3.0f; m1(2, 1) = 6.0f; m1(2, 2) = 9.0f;
	std::cout << m1 << std::endl;

	// comma initialization (row-major initialization)
	Matrix3f m2;
	m2 << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	std::cout << m1 << std::endl;

	// row, cols, size
	printf("%ix%i = %i\n", (int)m1.rows(), (int)m1.cols(), (int)m1.size());

	// DON'T DO THIS !!
	// m1 = m1.transpose();

	// NOTE: basic arithmetic operators, transpose(), conjugate() simply return a proxy object without doing the actual computation
	// Actual computation is happened with assignment operator.
	// use 'InPlace' version like below
	m1.transposeInPlace();

	// Matrix-matrix multiplication
	Matrix3f m3 = m1 * m2;
	std::cout << m3 << std::endl;

	// aliasing occured when m1 * m2 calculated
	m1 = m1 * m2;

	// no aliasing needed so that we can speed up with noalias() function
	m3.noalias() = m1 * m2;

	// block operation
	auto m1_block = m1.block<1, 1>(0, 0);
	auto col0_block = m1.col(0);
	auto row1_block = m1.row(1);

	// vector norm(), squaredNorm()
	Vector3f v1;
	v1 << 1, 2, 3;
	std::cout << v1 << std::endl;
	printf("v1.norm (length) = %f == %f\n", v1.norm(), sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]));
	printf("v1.squredNorm (squared length) = %f == %f\n", v1.squaredNorm(), v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);

	// LU decomposition
	PartialPivLU<Matrix3f> decomp(m3);
	Vector3f x = decomp.solve(v1);

	printf("is identity ? %s\n", m1 == Matrix3f::Identity() ? "true" : "false");
}

int main() {
	srand((unsigned int)time(0));

	TestEigen();

	getchar();

	return 0;
}
