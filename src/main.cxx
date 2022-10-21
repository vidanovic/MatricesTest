#include <chrono>
#include <iostream>
#include "SquareMatrixWCE.hxx"
#include "SquareMatrixSingleVector.hxx"

void fillMatrix(WCE::SquareMatrix& matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            matrix(i, j) = i + j;
        }
    }
}

void fillMatrix(SingleVector::SquareMatrix& matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            matrix(i, j) = i + j;
        }
    }
}

int main()
{
    constexpr size_t size = 145u;
    // Create SquareMatrix of size 145
    WCE::SquareMatrix matrix1(size);
    WCE::SquareMatrix matrix2(size);

    // Fill matrix with random values
    fillMatrix(matrix1, size);
    fillMatrix(matrix2, size);

    constexpr size_t iterations = 1000u;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Multiply matrices multiple times
    for (int i = 0; i < iterations; ++i)
    {
        matrix1 *= matrix2;
    }

    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    SingleVector::SquareMatrix matrix_v1(size);
    SingleVector::SquareMatrix matrix_v2(size);

    // Fill matrix with random values
    fillMatrix(matrix_v1, size);
    fillMatrix(matrix_v2, size);

    // Start timer
    start = std::chrono::high_resolution_clock::now();

    // Multiply matrices multiple times
    for (int i = 0; i < iterations; ++i)
    {
        matrix1 *= matrix2;
    }

    // Stop timer
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;


    return 0;
}