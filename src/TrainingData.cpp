#include "TrainingData.hpp"
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <iostream>

std::vector<std::string> split(std::string str, const std::string &delim)
{
    std::vector<std::string> result;

	size_t pos(0);
	while ((pos = str.find(delim)) != std::string::npos) {
		const auto token = str.substr(0, pos);
		result.push_back(token);
		str.erase(0, pos + delim.length());
	}
	result.push_back(str);

	return result;
}

std::string Test::toString() const 
{
    auto output = std::stringstream();
    output << "Test inputs: ";
    for (const auto &input: this->inputs) {
        output << input << ", ";
    }
    output << "Expected output: " << this->output;
    return output.str();
}

Test Test::createFromString(std::string test)
{
    Test output;

    auto testParts = split(test, ", ");
    output.output = atof(testParts.back().c_str());

    for (size_t i = 0; i < testParts.size() - 1; i++) {
        output.inputs.push_back(atof(testParts[i].c_str()));
    }
    return output;
}

TrainingData TrainingData::createFromFile(std::string fileName)
{
    std::vector<std::string>    testInput;
    std::ifstream               file;
    std::string                 line;
    TrainingData                output;
    
    file.open(fileName);
    while (std::getline(file, line)) {
        testInput.push_back(line);
    }
    file.close();

    const auto topologyStr = split(testInput[0], ", ");
    for (const auto &layerWidth : topologyStr) {
        output.topology.push_back(atoi(layerWidth.c_str()));
    }

    for (size_t i = 1; i < testInput.size(); i++) {
        const auto test = Test::createFromString(testInput[i]);
        output.trainingData.push_back(test);
    }

    return output;
}