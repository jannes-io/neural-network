#include "Neuron.hpp"
#include <cstdlib>
#include <cmath>
#include <iostream>

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

double randomWeight()
{
    return rand() / double(RAND_MAX);
}

Neuron::Neuron(unsigned int outputCount, unsigned int weightIndex)
{
    for (size_t conn = 0; conn < outputCount; conn++) {
        auto connection = Connection();
        connection.weight = randomWeight();
        this->connections.push_back(connection);
    }
    this->weightIndex = weightIndex;
}

Neuron::~Neuron() {}

void Neuron::feedForward(const Layer &previousLayer)
{
    double sum = 0.0;
    for (size_t n = 0; n < previousLayer.size(); n++) {
        sum += previousLayer[n].output *
            previousLayer[n].connections[this->weightIndex].weight;
    }

    this->output = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double target)
{
    double delta = target - this->output;
    this->gradient = delta * Neuron::transferFunctionDerivative(this->output);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = this->sumDOW(nextLayer);
    this->gradient = dow * Neuron::transferFunctionDerivative(this->output);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    for (size_t n = 0; n < nextLayer.size() - 1; n++) {
        sum += this->connections[n].weight * nextLayer[n].gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (size_t n = 0; n < prevLayer.size(); n++) {
        auto &neuron = prevLayer[n];

        double oldDeltaWeight = neuron.connections[this->weightIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.output * this->gradient + alpha * oldDeltaWeight;

        neuron.connections[this->weightIndex].deltaWeight = newDeltaWeight;
        neuron.connections[this->weightIndex].weight += newDeltaWeight;
    }
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}