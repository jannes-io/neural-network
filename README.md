# Simple Neural Network

This is a simple DFF neural network implementation.

## Usage

### Windows
```
make; ./build/neural-net.exe ../training/<trainingfile>
```
(or drag and drop your trainingfile onto the .exe)

### macOS/Linux
```
make && ./build/neural-net ../training/<trainingfile>
```

### Preparing training data

Ex. for boolean logic AND:
```
2, 3, 1
1, 1, 1
0, 1, 0
0, 0, 0
1, 1, 1
1, 0, 0
0, 0, 0
```

The first line, `2, 3, 1` denotes the topography of the network. In this case the network will
consist of 3 layers with 2 input neurons, 3 neurons on a hidden layer, and 1 output neuron.

All other lines are tests, with n inputs and m outputs where n is equal to the amount of
input nodes, and m equal to the amount of output nodes. Always delimited by `, ` (comma, space).