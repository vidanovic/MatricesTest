#include <stdexcept>
#include <cassert>
#include <cmath>

#include "SquareMatrixSingleVector.hxx"

namespace SingleVector
{
    SquareMatrix::SquareMatrix(const std::size_t tSize) :
        m_size(tSize),
        m_Matrix(tSize * tSize, 0)
    {}

    std::size_t SquareMatrix::size() const
    {
        return m_size;
    }

    void SquareMatrix::setZeros()
    {
        for(auto & val : m_Matrix)
        {
            val = 0;
        }
    }

    void SquareMatrix::setIdentity()
    {
        setZeros();
        for(auto i = 0u; i < m_size; ++i)
        {
            m_Matrix[i + m_size * i] = 1;
        }
    }

    void SquareMatrix::setDiagonal(const std::vector<double> & tInput)
    {
        if(tInput.size() != m_size)
        {
            throw std::runtime_error("Matrix and vector must be same size.");
        }

        setZeros();
        for(auto i = 0u; i < m_size; ++i)
        {
            m_Matrix[i + m_size * i] = tInput[i];
        }
    }


    SquareMatrix SquareMatrix::inverse() const
    {
        // return LU decomposed matrix of current matrix
        auto aLu(LU());

        // find the inverse
        SquareMatrix invMat(m_size);
        std::vector<double> d(m_size);
        std::vector<double> y(m_size);

        const auto size(m_size - 1);

        for(auto m = 0u; m <= size; ++m)
        {
            fill(d.begin(), d.end(), 0);
            fill(y.begin(), y.end(), 0);
            d[m] = 1;
            for(auto i = 0; i <= int(size); ++i)
            {
                double x = 0;
                for(auto j = 0; j <= i - 1; ++j)
                {
                    x = x + aLu(size_t(i), size_t(j)) * y[j];
                }
                y[i] = (d[i] - x);
            }

            for(auto i = int(size); i >= 0; --i)
            {
                auto x = 0.0;
                for(auto j = i + 1; j <= int(size); ++j)
                {
                    x = x + aLu(size_t(i), size_t(j)) * invMat(size_t(j), size_t(m));
                }
                invMat(size_t(i), size_t(m)) = (y[i] - x) / aLu(size_t(i), size_t(i));
            }
        }

        return invMat;
    }

    double SquareMatrix::operator()(const std::size_t i, const std::size_t j) const
    {
        return m_Matrix[i + m_size * j];
    }

    double & SquareMatrix::operator()(const std::size_t i, const std::size_t j)
    {
        return m_Matrix[i + m_size * j];
    }

    SquareMatrix SquareMatrix::LU() const
    {
        SquareMatrix D(m_size);
        D = *this;

        for(auto k = 0u; k <= m_size - 2; ++k)
        {
            for(auto j = k + 1; j <= m_size - 1; ++j)
            {
                const auto x = D(j, k) / D(k, k);
                for(auto i = k; i <= m_size - 1; ++i)
                {
                    D(j, i) = D(j, i) - x * D(k, i);
                }
                D(j, k) = x;
            }
        }

        return D;
    }

    std::vector<double> SquareMatrix::checkSingularity() const
    {
        std::vector<double> vv;

        for(auto i = 0u; i < m_size; ++i)
        {
            auto aamax = 0.0;
            for(size_t j = 0; j < m_size; ++j)
            {
                const auto absCellValue = std::abs(m_Matrix[i + m_size * j]);
                if(absCellValue > aamax)
                {
                    aamax = absCellValue;
                }
            }
            if(aamax == 0)
            {
                assert(aamax != 0);
            }
            vv.push_back(1 / aamax);
        }

        return vv;
    }

    std::vector<size_t> SquareMatrix::makeUpperTriangular()
    {
        const auto TINY(1e-20);

        std::vector<size_t> index(m_size);

        std::vector<double> vv = checkSingularity();

        auto d = 1;

        for(auto j = 0u; j < m_size; ++j)
        {
            for(auto i = 0; i <= int(j - 1); ++i)
            {
                auto sum = m_Matrix[i + m_size * j];
                for(auto k = 0; k <= i - 1; ++k)
                {
                    sum = sum - m_Matrix[i + k * m_size] * m_Matrix[k + j * m_size];
                }
                m_Matrix[i + m_size * j] = sum;
            }

            auto aamax = 0.0;
            auto imax = 0;

            for(auto i = j; i < m_size; ++i)
            {
                auto sum = m_Matrix[i + m_size * j];
                for(auto k = 0; k <= int(j - 1); ++k)
                {
                    sum = sum - m_Matrix[i + m_size * k] * m_Matrix[k + m_size * j];
                }
                m_Matrix[i + m_size * j] = sum;
                const auto dum = vv[i] * std::abs(sum);
                if(dum >= aamax)
                {
                    imax = i;
                    aamax = dum;
                }
            }

            if(int(j) != imax)
            {
                for(auto k = 0u; k < m_size; ++k)
                {
                    const auto dum = m_Matrix[imax + m_size * k];
                    m_Matrix[imax + m_size * k] = m_Matrix[j + m_size * k];
                    m_Matrix[j + m_size * k] = dum;
                }   // k
                d = -d;
                vv[imax] = vv[j];
            }
            index[j] = imax;
            if(m_Matrix[j + m_size * j] == 0.0)
            {
                m_Matrix[j + m_size * j] = TINY;
            }
            if(j != (m_size - 1))
            {
                const auto dum = 1.0 / m_Matrix[j + m_size * j];
                for(auto i = j + 1; i < m_size; ++i)
                {
                    m_Matrix[i + m_size * j] = m_Matrix[i + m_size * j] * dum;
                }   // i
            }
        }

        return index;
    }

    SquareMatrix operator*(const SquareMatrix & first, const SquareMatrix & second)
    {
        if(first.size() != second.size())
        {
            throw std::runtime_error("Matrices must be identical in size.");
        }

        SquareMatrix aMatrix{first.size()};

        for(size_t i = 0; i < aMatrix.size(); ++i)
        {
            for(size_t k = 0; k < aMatrix.size(); ++k)
            {
                for(size_t j = 0; j < aMatrix.size(); ++j)
                {
                    aMatrix(i, j) += first(i, k) * second(k, j);
                }
            }
        }

        return aMatrix;
    }

    SquareMatrix operator*=(SquareMatrix & first, const SquareMatrix & second)
    {
        first = first * second;
        return first;
    }

    SquareMatrix operator+(const SquareMatrix & first, const SquareMatrix & second)
    {
        if(first.size() != second.size())
        {
            throw std::runtime_error("Matrices must be identical in size.");
        }

        SquareMatrix aMatrix{first.size()};
        for(size_t i = 0; i < aMatrix.size(); ++i)
        {
            for(size_t j = 0; j < aMatrix.size(); ++j)
            {
                aMatrix(i, j) = first(i, j) + second(i, j);
            }
        }

        return aMatrix;
    }

    SquareMatrix operator+=(SquareMatrix & first, const SquareMatrix & second)
    {
        first = first + second;
        return first;
    }

    SquareMatrix operator-(const SquareMatrix & first, const SquareMatrix & second)
    {
        if(first.size() != second.size())
        {
            throw std::runtime_error("Matrices must be identical in size.");
        }

        SquareMatrix aMatrix(first.size());
        for(size_t i = 0; i < aMatrix.size(); ++i)
        {
            for(size_t j = 0; j < aMatrix.size(); ++j)
            {
                aMatrix(i, j) = first(i, j) - second(i, j);
            }
        }

        return aMatrix;
    }

    SquareMatrix operator-=(SquareMatrix & first, const SquareMatrix & second)
    {
        first = first - second;
        return first;
    }

    SquareMatrix SquareMatrix::mmultRows(const std::vector<double> & tInput)
    {
        if(m_size != tInput.size())
        {
            throw std::runtime_error("Vector and matrix do not have same size.");
        }

        SquareMatrix res{m_size};
        for(auto i = 0u; i < m_size; ++i)
        {
            for(auto j = 0u; j < m_size; ++j)
            {
                res(j, i) = m_Matrix[j + m_size * i] * tInput[i];
            }
        }

        return res;
    }


    std::vector<double> operator*(const std::vector<double> & first, const SquareMatrix & second)
    {
        if(first.size() != second.size())
        {
            throw std::runtime_error("Vector and matrix do not have same size.");
        }

        std::vector<double> res(first.size(), 0);

        for(auto i = 0u; i < first.size(); ++i)
        {
            for(auto j = 0u; j < first.size(); ++j)
            {
                res[i] += first[j] * second(j, i);
            }
        }

        return res;
    }

    std::vector<double> operator*(const SquareMatrix & first, const std::vector<double> & second)
    {
        if(first.size() != second.size())
        {
            throw std::runtime_error("Vector and matrix do not have same size.");
        }

        std::vector<double> res(second.size(), 0);

        for(auto i = 0u; i < second.size(); ++i)
        {
            for(auto j = 0u; j < second.size(); ++j)
            {
                res[i] += second[j] * first(i, j);
            }
        }

        return res;
    }

    SquareMatrix multiplyWithDiagonalMatrix(const std::vector<double> & tInput,
                                            const SquareMatrix & tMatrix)
    {
        SquareMatrix res{tInput.size()};
        for(auto i = 0u; i < tInput.size(); ++i)
        {
            for(auto j = 0u; j < tInput.size(); ++j)
            {
                res(i, j) = tMatrix(i, j) * tInput[i];
            }
        }
        return res;
    }

    SquareMatrix multiplyWithDiagonalMatrix(const SquareMatrix & tMatrix,
                                                                const std::vector<double> & tInput)
    {
        SquareMatrix res{tInput.size()};
        for(auto i = 0u; i < tInput.size(); ++i)
        {
            for(auto j = 0u; j < tInput.size(); ++j)
            {
                res(i, j) = tMatrix(i, j) * tInput[j];
            }
        }
        return res;
    }


}   // namespace FenestrationCommon
