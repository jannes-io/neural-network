#include <iostream>
#include "Net.hpp"
#include "TrainingData.hpp"

int main() {
    const auto trainingData = TrainingData::createFromFile("../training/addition.txt");
    Net neuralNet(trainingData.topology);

    int passes = 0;
    for (const auto &test : trainingData.trainingData) {
        std::cout << ++passes << ": " << test.toString() << '\n';
        neuralNet.feedForward(test.inputs);

        const auto result = neuralNet.getResults();
        std::cout << "Guessed output: ";
        for (const auto &answer : result) {
            std::cout << answer << " ";
        }

        neuralNet.backProp(std::vector<double>{test.output});

        std::cout << "\nAverage Error: " << neuralNet.averageError << "\n";
        std::cout << std::endl;
    }

    return 0;
}