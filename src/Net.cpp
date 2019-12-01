#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include "Neuron.hpp"
#include "Net.hpp"

double Net::errorSmoothingFactor = 100.0;

Net::Net(const std::vector<unsigned int> &topology)
{
    const auto netSize = topology.size();
    for (size_t layer = 0; layer < netSize; layer++) {
        this->layers.push_back(Layer());
        const auto outputCount = layer == netSize - 1
            ? 0
            : topology[layer + 1];

        // 1 extra Neuron per layer for bias
        for (size_t i = 0; i <= topology[layer]; i++) {
            this->layers.back().push_back(Neuron(outputCount, i));
            std::cout << "Layer: " << layer << " Neuron: " << i << " created" << std::endl;
        }
        this->layers.back().back().output = 1.0;
    }
}

Net::~Net() {}

void Net::feedForward(const std::vector<double> &trainingData)
{
    assert(trainingData.size() == this->layers[0].size() - 1);
    for (size_t i = 0; i < trainingData.size(); i++) {
        this->layers[0][i].output = trainingData[i];
    }

    for (size_t l = 1; l < this->layers.size(); l++) {
        const auto &previousLayer = this->layers[l - 1];
        for (size_t n = 0; n < this->layers[l].size() - 1; n++) {
            this->layers[l][n].feedForward(previousLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &expectedResults)
{
    // RMS (Root Mean Square Error)
    auto &outputLayer = this->layers.back();
    this->error = 0.0;
    for (size_t n = 0; n < outputLayer.size() - 1; n++) {
        double delta = expectedResults[n] - outputLayer[n].output;
        this->error += delta * delta;
    }
    // Calculate RMS
    this->error /= outputLayer.size() - 1;
    this->error = sqrt(this->error);

    this->averageError = (this->averageError * this->errorSmoothingFactor + this->error)
        / (this->errorSmoothingFactor + 1.0);

    // Calculate output gradients
    for (size_t n = 0; n < outputLayer.size() - 1; n++) {
        outputLayer[n].calcOutputGradients(expectedResults[n]);
    }

    // Calculate hidden output gradients
    for (size_t l = this->layers.size() - 2; l > 0; l--) {
        auto &hiddenLayer = this->layers[l];
        auto &nextLayer = this->layers[l + 1];

        for (size_t n = 0; n < hiddenLayer.size(); n++) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // Update connection weights
    for (size_t l = this->layers.size() - 1; l > 0; l--) {
        auto &layer = this->layers[l];
        auto &prevLayer = this->layers[l - 1];

        for (size_t n = 0; n < layer.size() - 1; n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

std::vector<double> Net::getResults()
{
    std::vector<double> results;
    for (size_t n = 0; n < this->layers.back().size() - 1; n++) {
        results.push_back(this->layers.back()[n].output);
    }
    return results;
}

