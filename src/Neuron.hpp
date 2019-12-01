#pragma once

#include <vector>

struct Connection
{
    double weight;
    double deltaWeight;
};

struct Neuron;

typedef std::vector<Neuron> Layer;

struct Neuron
{
    double output;
    double gradient;
    unsigned int weightIndex;
    std::vector<Connection> connections;

    static double eta;
    static double alpha;

    Neuron(unsigned int outputCount, unsigned int weightIndex);
    ~Neuron();

    void feedForward(const Layer &previousLayer);
    void calcOutputGradients(double target);
    void calcHiddenGradients(const Layer &nextLayer);
    double sumDOW(const Layer &nextLayer) const;
    void updateInputWeights(Layer &prevLayer);

    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
};
