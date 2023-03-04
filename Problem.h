#if !defined(PROBLEM_H)
#define PROBLEM_H

#include <vector>
#include <cstddef>
#include <string>
#include <chrono>
#include <fstream>

using std::vector, std::size_t;
using duration = std::chrono::duration<double>;

class Problem
{

public:
    Problem(size_t _size);
    ~Problem();

    void decompose();
    void calculateDeterminant();

    void setNumThreads(size_t _numThreads);
    void setExperimentNum(size_t _experimentNum) { m_experimentNum = _experimentNum; }

    void check();
    void reset();
    void print(bool _printMatrices);

    void save();

private:
    size_t m_size;

    vector<vector<double>> m_A;
    vector<vector<double>> m_L;
    vector<vector<double>> m_U;

    vector<vector<double>> m_newA;

    double m_residual;
    double m_determinant;

    duration m_timeForDecomposition;
    duration m_timeForDeterminant;
    duration m_timeForCheck;

    size_t m_numThreads;
    size_t m_experimentNum;

    std::string m_filename;
    std::ofstream m_file;

    static void printMatrix(vector<vector<double>> &_matrix, std::string _name);
};

#endif // PROBLEM_H