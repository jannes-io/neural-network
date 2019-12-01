#pragma once

#include <vector>
#include "Neuron.hpp"

struct Net
{
    double error = 0.0;
    double averageError = 0.0;
    static double errorSmoothingFactor;
    std::vector<Layer> layers;

    Net(const std::vector<unsigned int> &topology);
    ~Net();

    void feedForward(const std::vector<double> &trainingData);
    void backProp(const std::vector<double> &trainingResults);
    std::vector<double> getResults();
};