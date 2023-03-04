#include "Problem.h"

#include <string>
#include <iostream>

int main(int argc, char const *argv[])
{
    bool verbose{false};
    bool verboseForMatrix{false};
    int maxNumThreads{1};
    size_t sizeOfMatrix{100};

    std::cout << std::endl;

    for (size_t argNum = 1; argNum < argc; argNum++)
    {
        if (std::string(argv[argNum]) == "-verbose")
            verbose = true;
        else if (std::string(argv[argNum]) == "-printMatrices")
            verboseForMatrix = true;
        else if (std::string(argv[argNum]) == "-threads")
            maxNumThreads = std::stoi(argv[++argNum]);
        else if (std::string(argv[argNum]) == "-size")
            sizeOfMatrix = std::stoi(argv[++argNum]);
        else if (std::string(argv[argNum]) == "-help")
        {
            std::cout << "There are some keys to manage program." << std::endl
                      << "-help -- show this message" << std::endl
                      << "-verbose -- print in console some results (residual, time, etc.)" << std::endl
                      << "-printMatrices -- print matrices in console" << std::endl
                      << "-threads <num> -- set maximum number of threads to use" << std::endl
                      << "-size <num> -- set size of matrix to decompose" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Use -help key to get usage instructions." << std::endl;
            return 0;
        }
    }

    Problem problem(sizeOfMatrix);

    for (int numOfThreads{1}; numOfThreads <= maxNumThreads; numOfThreads++)
    {
        problem.setNumThreads(numOfThreads);
        for (int expertimentNum{1}; expertimentNum <= 5; expertimentNum++)
        {
            //problem.setExperimentNum(expertimentNum);
            problem.decompose();
            //problem.check();
            problem.calculateDeterminant();

            if (verbose)
                problem.print(verboseForMatrix);
            
            problem.save();

            problem.reset();
        }
    }
    return 0;
}
