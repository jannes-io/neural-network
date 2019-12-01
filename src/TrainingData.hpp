#pragma once

#include <vector>
#include <string>

struct Test {
    std::vector<double> inputs;
    double output;

    std::string toString() const;
    static Test createFromString(std::string test);
};

typedef std::vector<unsigned int> Topology;

struct TrainingData {
    Topology topology;
    std::vector<Test> trainingData;

    static TrainingData createFromFile(std::string fileName);
};