#pragma once

#include <cstddef>
#include <vector>

#pragma warning(push, 0)
//#include <Eigen/Core>
#include <Eigen/SparseCore>
#pragma warning(pop)

namespace HTF
{
    //! \brief It is build to serve purposes for finite elements.
    //!
    //! Class is tied to Eigen package that is used in background to perform matrix operations.
    class SquareMatrix
    {
    public:
    	//! Construction of empty square matrix with given size.
        explicit SquareMatrix(
        	std::size_t m_size //!< Matrix size
        	);

        //! Creating matrix from initializer list.
        explicit SquareMatrix(const std::initializer_list<std::vector<double>> & tInput);

        //! Creating matrix from standard vector of vectors.
        explicit SquareMatrix(const std::vector<std::vector<double>> & tInput);

		//! Creating matrix from standard vector of vectors.
        explicit SquareMatrix(const std::vector<std::vector<double>> && tInput);

        //! Returns matrix size
        std::size_t size() const;

        //! Set all elements of the matrix to zeros.
        void setZeros();

        //! Operator to access element of matrix
        double operator()(std::size_t i, std::size_t j) const;

        //! Operator to access element of matrix
        double & operator()(std::size_t i, std::size_t j);

        //! Every row in matrix is multiplied with values in vector (element by element).
        SquareMatrix mmultRows(const std::vector<double> & tInput) const;

        //! Operator to multiply every element of matrix with single double value.
        SquareMatrix operator*(double value) const;

		//! Operator to multiply every element of matrix with single double value.
        friend SquareMatrix operator*(double value, const SquareMatrix & other);

        //! Two matrices multiplication
        friend SquareMatrix operator*(const SquareMatrix & first, const SquareMatrix & second);

        //! *= operator overload for matrix
        SquareMatrix & operator*=(const SquareMatrix & other);

        //! Addition of two matrices
        SquareMatrix & operator+(const SquareMatrix & other);

        //! *= operator overload
        SquareMatrix & operator+=(const SquareMatrix & other);

        //! Two matrices subtraction
        SquareMatrix & operator-(const SquareMatrix & other);

        //! -= operator overload
        SquareMatrix & operator-=(const SquareMatrix & other);

        //! Multiplication between matrix and vector (M x V)
        std::vector<double> operator*(const std::vector<double> & tVec) const;

        //! Multiplication between vector and matrix (V x M)
        friend std::vector<double> operator*(const std::vector<double> & first,
                                             const SquareMatrix & second);

        //! Convert matrix to standard vector of vectors. Function is only used for debugging
        //! purposes.
        std::vector<std::vector<double>> toVector() const;

    private:
        //! Creation of square matrix from sparse matrix.
        explicit SquareMatrix(Eigen::MatrixXd && tMatrix);
        std::size_t m_size;
        Eigen::MatrixXd m_Matrix;
        // Eigen::MatrixXd m_Matrix;
    };

}   // namespace HygroThermFEM
