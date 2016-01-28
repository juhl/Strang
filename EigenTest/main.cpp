#include <iostream>
#include <algorithm>
#include <time.h>
#include "../Eigen/Dense"

using namespace Eigen;

void TestBasic() {
	// fixed size matrix type
	printf("Matrix<float, 2, 2> == Matrix2f ? %s\n", typeid(Matrix<float, 2, 2>) == typeid(Matrix2f) ? "yes" : "no");
	printf("Matrix<float, 3, 3> == Matrix3f ? %s\n", typeid(Matrix<float, 3, 3>) == typeid(Matrix3f) ? "yes" : "no");
	printf("Matrix<float, 4, 4> == Matrix4f ? %s\n", typeid(Matrix<float, 4, 4>) == typeid(Matrix4f) ? "yes" : "no");
	printf("Matrix<float, 3, 1> == Vector3f ? %s\n", typeid(Matrix<float, 3, 1>) == typeid(Vector3f) ? "yes" : "no");
	printf("Matrix<float, 1, 3> == RowVector3f ? %s\n", typeid(Matrix<float, 1, 3>) == typeid(RowVector3f) ? "yes" : "no");

	// dynamic size matrix type
	printf("Matrix<float, Dynamic, Dynamic> == MatrixXf ? %s\n", typeid(Matrix<float, Dynamic, Dynamic>) == typeid(MatrixXf) ? "yes" : "no");
	printf("Matrix<float, Dynamic, 1> == VectorXf ? %s\n", typeid(Matrix<float, Dynamic, 1>) == typeid(VectorXf) ? "yes" : "no");
	printf("Matrix<float, 1, Dynamic> == RowVectorXf ? %s\n", typeid(Matrix<float, 1, Dynamic>) == typeid(RowVectorXf) ? "yes" : "no");

	// coefficient accessors (matrix element is called 'coefficient' in Eigen)
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
}

void TestLinearSystemSolver() {
	Matrix3f A = Matrix3f::Random(3, 3);
	A = A.transpose() * A; // make A to symmetric positive (semi) definite
	Vector3f b = Vector3f::Random(3);
	Vector3f x;
	float relativeError; // check the solution really exist with this

	std::cout << "=============================" << std::endl;
	std::cout << "Testing linear system solvers" << std::endl;
	std::cout << "=============================" << std::endl;

	// A must be invertible
	PartialPivLU<Matrix3f> partialPivLU(A);
	x = partialPivLU.solve(b);
	relativeError = (A * x - b).norm() / b.norm();
	if (relativeError < 0.00001) {
		std::cout << "Solution using partial pivoting LU = " << std::endl << x.transpose() << std::endl;
	}

	FullPivLU<Matrix3f> fullPivLU(A);
	x = fullPivLU.solve(b);
	relativeError = (A * x - b).norm() / b.norm();
	if (relativeError < 0.00001) {
		std::cout << "Solution using full pivoting LU = " << std::endl << x.transpose() << std::endl;
	}

	HouseholderQR<Matrix3f> householderQR(A);
	x = householderQR.solve(b);
	relativeError = (A * x - b).norm() / b.norm();
	if (relativeError < 0.00001) {
		std::cout << "Solution using Householder QR = " << std::endl << x.transpose() << std::endl;
	}

	ColPivHouseholderQR<Matrix3f> colPivHouseholderQR(A);
	x = colPivHouseholderQR.solve(b);
	relativeError = (A * x - b).norm() / b.norm();
	if (relativeError < 0.00001) {
		std::cout << "Solution using column pivoting Householder QR = " << std::endl << x.transpose() << std::endl;
	}

	FullPivHouseholderQR<Matrix3f> fullPivHouseholderQR(A);
	x = fullPivHouseholderQR.solve(b);
	relativeError = (A * x - b).norm() / b.norm();
	if (relativeError < 0.00001) {
		std::cout << "Solution using full pivoting Householder QR = " << std::endl << x.transpose() << std::endl;
	}

	// A must be positive definite
	LLT<Matrix3f> llt(A);
	x = llt.solve(b);
	relativeError = (A * x - b).norm() / b.norm();
	if (relativeError < 0.00001) {
		std::cout << "Solution using Cholesky decomposition = " << std::endl << x.transpose() << std::endl;
	}

	// A must not be indefinite
	LDLT<Matrix3f> ldlt(A);
	x = ldlt.solve(b);
	relativeError = (A * x - b).norm() / b.norm();
	if (relativeError < 0.00001) {
		std::cout << "Solution using LDLT decomposition = " << std::endl << x.transpose() << std::endl;
	}
}

void TestLeastSquares() {
	MatrixXf A = MatrixXf::Random(10, 2);
	VectorXf b = VectorXf::Random(10);
	Vector2f x;

	std::cout << "=============================" << std::endl;
	std::cout << "Testing least squares solvers" << std::endl;
	std::cout << "=============================" << std::endl;

	x = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
	std::cout << "Solution using Jacobi SVD = " << x.transpose() << std::endl;

	x = A.colPivHouseholderQr().solve(b);
	std::cout << "Solution using column pivoting Householder QR = " << x.transpose() << std::endl;

	// If the matrix A is ill-conditioned, then this is not a good method
	x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
	std::cout << "Solution using normal equation = " << x.transpose() << std::endl;
}

void TestEigenSolver() {
	Matrix3f A = Matrix3f::Random(3, 3);
	A = A.transpose() * A; // make A to symmetric positive (semi) definite
	Vector3f b = Vector3f::Random(3);
	Vector3f x;

	std::cout << "=====================" << std::endl;
	std::cout << "Testing Eigen solvers" << std::endl;
	std::cout << "=====================" << std::endl;

	// A must be (real) symmetric 
	SelfAdjointEigenSolver<Matrix3f> selfAdjointEigenSolver(A);
	std::cout << "Eigenvalues = " << selfAdjointEigenSolver.eigenvalues().transpose() << std::endl;
	std::cout << "Eigenvectors = " << selfAdjointEigenSolver.eigenvectors() << std::endl;

	// NOTE: sort by yourself
	EigenSolver<Matrix3f> eigenSolver(A);
	std::cout << "Eigenvalues = " << eigenSolver.eigenvalues().transpose() << std::endl;
	std::cout << "Eigenvectors = " << eigenSolver.eigenvectors() << std::endl;
}

void TestSVD() {
	MatrixXf A = MatrixXf::Random(4, 3);

	std::cout << "===========" << std::endl;
	std::cout << "Testing SVD" << std::endl;
	std::cout << "===========" << std::endl;

	JacobiSVD<MatrixXf> jacobiSVD(A, ComputeThinU | ComputeThinV);
	std::cout << "A = " << A << std::endl;
	std::cout << "U = " << jacobiSVD.matrixU() << std::endl;
	std::cout << "Sigma = " << jacobiSVD.singularValues().transpose() << std::endl;
	std::cout << "V = " << jacobiSVD.matrixV() << std::endl;
}

int main() {
	srand((unsigned int)time(0));

	TestBasic();

	TestLinearSystemSolver();

	TestLeastSquares();

	TestEigenSolver();

	TestSVD();

	getchar();

	return 0;
}
