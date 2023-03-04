#include "Problem.h"

#include <random>
#include <iostream>
#include <cmath>
#include <omp.h>

#define STRINGIZE(x) #x
#define GET_MACRO_VALUE(x) STRINGIZE(x)

#ifdef SCHEDULE
#if SCHEDULE == 0
#undef SCHEDULE
#define SCHEDULE static
#elif SCHEDULE == 1
#undef SCHEDULE
#define SCHEDULE dynamic
#elif SCHEDULE == 2
#undef SCHEDULE
#define SCHEDULE guided
#endif
#endif // SCHEDULE

Problem::Problem(size_t _size)
    : m_size{_size},
      m_A{vector<vector<double>>(m_size, vector<double>(m_size, 0.0))},
      m_L{vector<vector<double>>(m_size, vector<double>(m_size, 0.0))},
      m_U{vector<vector<double>>(m_size, vector<double>(m_size, 0.0))},
      m_newA{vector<vector<double>>(m_size, vector<double>(m_size, 0.0))},
      m_residual{0.0}, m_determinant{0.0},
      m_timeForDecomposition{0.0}, m_timeForDeterminant{0.0},
      m_numThreads{1}, m_experimentNum{1}
{
    // Special initialization of A matrix to control
    // magnitude of its determinant
    std::uniform_real_distribution<double> diagDistribution(0.9, 1.1);
    std::random_device rd;
    std::mt19937 mersenne(rd());
    for (size_t row{0}; row < m_size; row++)
    {
        for (size_t col{0}; col < m_size; col++)
        {
            row == col ? m_A[row][col] = diagDistribution(mersenne) : m_A[row][col] = 1.0 / (row + 2);
        }
    }

    m_filename = "./Results/" GET_MACRO_VALUE(SCHEDULE);

#if LOCK == 1
    m_filename += "lock";
#endif

    m_filename += std::to_string(m_size) + ".txt";

    m_file.open(m_filename, std::ios::trunc);
    m_file.close();
}

Problem::~Problem(){};

void Problem::decompose()
{
    using std::chrono::high_resolution_clock,
        std::chrono::duration_cast,
        std::chrono::milliseconds;

    // Start point for timer
    auto timeStart = high_resolution_clock::now();

    double sum = 0.0;
    // LU decomposition

#if LOCK == 0 // NO LOCK
#pragma omp parallel num_threads(m_numThreads) shared(m_A, m_L, m_U)
#else  // init lock
    omp_lock_t lock;
    omp_init_lock(&lock);
#endif // NO LOCK

    for (size_t row{0}; row < m_size; row++)
    {
        m_L[row][row] = 1.0;

#if LOCK == 0 // NO LOCK
#pragma omp for schedule(SCHEDULE) reduction(+ \
                                             : sum)
#endif // NO LOCK

        for (size_t col = row; col < m_size; col++)
        {
            sum = 0.0;
#if LOCK == 1 // LOCK
#pragma omp parallel for num_threads(m_numThreads) shared(m_A, m_L, m_U, sum, lock) schedule(SCHEDULE)
#endif
            for (size_t i = 0; i < row; i++)
            {

#if LOCK == 1 // LOCK
                omp_set_lock(&lock);
                sum += m_L[row][i] * m_U[i][col];
                omp_unset_lock(&lock);

#else // NO LOCK
                sum += m_L[row][i] * m_U[i][col];
#endif
            }
            m_U[row][col] = m_A[row][col] - sum;
        }

#if LOCK == 0 // NO LOCK
#pragma omp for schedule(SCHEDULE) reduction(+ \
                                             : sum)
#endif // NO LOCK

        for (int col = row + 1; col < m_size; col++)
        {
            sum = 0.0;
#if LOCK == 1 // LOCK
#pragma omp parallel for num_threads(m_numThreads) shared(m_A, m_L, m_U, sum, lock) schedule(SCHEDULE)
#endif
            for (int i = 0; i < row; i++)
            {

#if LOCK == 1 // LOCK
                omp_set_lock(&lock);
                sum += m_L[col][i] * m_U[i][row];
                omp_unset_lock(&lock);

#else // NO LOCK
                sum += m_L[col][i] * m_U[i][row];
#endif
            }
            m_L[col][row] = (m_A[col][row] - sum) / m_U[row][row];
        }
    }

#if LOCK == 1 // LOCK for lock destruction
    omp_destroy_lock(&lock);
#endif

    // Stop point for timer
    auto timeStop = high_resolution_clock::now();

    // Elapsed time for LU decomposition
    m_timeForDecomposition = duration_cast<milliseconds>(timeStop - timeStart);
}

void Problem::check()
{
    // Check correctness with LU multiplication
    for (size_t row = 0; row < m_size; row++)
    {
        for (size_t col = 0; col < m_size; col++)
        {
            for (size_t i = 0; i < m_size; i++)
            {
                m_newA[row][col] += m_L[row][i] * m_U[i][col];
            }
        }
    }

    // Then calculate norm of residual as sum of squared difference of A elements
    // normalized by number of elements.

    double sum{0.0};
    for (size_t row{0}; row < m_size; row++)
    {
        for (size_t col{0}; col < m_size; col++)
        {
            sum += (m_A[row][col] - m_newA[row][col]) * (m_A[row][col] - m_newA[row][col]) / m_size / m_size;
        }
    }
    m_residual = sqrt(sum);
}

void Problem::calculateDeterminant()
{
    using std::chrono::high_resolution_clock,
        std::chrono::duration_cast,
        std::chrono::milliseconds;
    // Start point for timer

    auto timeStart = high_resolution_clock::now();

    // Calculating determinant of A as determinant of U
    m_determinant = 1.0;

#if LOCK == 0
#pragma omp parallel num_threads(m_numThreads) shared(m_U)
#pragma omp for schedule(SCHEDULE) reduction(* \
                                             : m_determinant)
#elif LOCK == 1
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel num_threads(m_numThreads) shared(m_determinant, m_U, lock)
#pragma omp parallel for schedule(SCHEDULE)
#endif
    for (size_t row = 0; row < m_size; row++)
    {
#if LOCK == 1
        omp_set_lock(&lock);
        m_determinant *= m_U[row][row];
        omp_unset_lock(&lock);
#elif LOCK == 0
        m_determinant *= m_U[row][row];
#endif
    }

#if LOCK == 1
    omp_destroy_lock(&lock);
#endif

    auto timeStop = high_resolution_clock::now();

    m_timeForDeterminant = duration_cast<milliseconds>(timeStop - timeStart);
}

void Problem::reset()
{
    // Reset data to initial state

    m_L.assign(m_size, vector<double>(m_size, 0.0));
    m_U.assign(m_size, vector<double>(m_size, 0.0));
    m_newA.assign(m_size, vector<double>(m_size, 0.0));

    m_residual = 0.0;
    m_determinant = 0.0;

    m_timeForDecomposition = duration::zero();
    m_timeForDeterminant = duration::zero();
}

void Problem::print(bool _printMatrices)
{
    // Console output
    if (_printMatrices)
    {
        Problem::printMatrix(m_A, "Origin Matrix");
        Problem::printMatrix(m_L, "L Matrix");
        Problem::printMatrix(m_U, "U Matrix");
        Problem::printMatrix(m_newA, "Check Matrix");
    }

    std::cout << "Residual is " << m_residual << std::endl;
    std::cout << "Determinamt is " << m_determinant << std::endl;

    std::cout << "Elapsed time for Decomposition: " << m_timeForDecomposition.count() << " s." << std::endl;
    std::cout << "Elapsed time for Determinant: " << m_timeForDeterminant.count() << " s." << std::endl;

    std::cout << std::endl;
}

void Problem::printMatrix(vector<vector<double>> &_matrix, std::string _name)
{
    // Print single matrix in console
    std::cout << _name << ':' << std::endl
              << std::endl;

    for (vector<double> &row : _matrix)
    {
        for (double &element : row)
        {
            std::cout << element << '\t';
        }
        std::cout << std::endl;
    }

    std::cout << "----------" << std::endl;
}

void Problem::setNumThreads(size_t _numThreads)
{
    m_numThreads = _numThreads;
}

void Problem::save()
{
    m_file.open(m_filename, std::ios::app);

    m_file << m_numThreads << '\t'
           << m_timeForDecomposition.count() << '\t'
           << m_timeForDeterminant.count() << '\t'
           << m_residual << '\t'
           << m_determinant << std::endl;

    m_file.close();
}