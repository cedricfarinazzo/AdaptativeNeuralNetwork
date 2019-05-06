# AdaptativeNeuralNetwork ANN (0.1.1.0)

### Build status

| Branch | Status |
|--------|--------|
|        |        |
| Master | [![pipeline status](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/master/pipeline.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/master)       |
|        |        |
| Develop| [![pipeline status](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/develop/pipeline.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/develop)       |


### Coverage report

| Branch | Status |
|--------|--------|
|        |        |
| Master | [![coverage report](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/master/coverage.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/master)       |
|        |        |
| Develop| [![coverage report](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/develop/coverage.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/develop)       |


## Requirements
- gcc or clang
- make
- [cmake](https://cmake.org/)
- [criterion](https://criterion.readthedocs.io/en/master/) (test only)


## How to build

- Clone this repo
- Run the following command
```
mkdir build && cd build && cmake .. && make
```


## Run test

```
make check
```


## Coverage report

```
make coverage
```


## Installation

```
sudo make install
```


## AdaptativeNeuralNetwork

A static library containing multiple neural network models written in C


## Examples

See some examples [here](https://github.com/cedricfarinazzo/ANNExample)


## Contribute

- Fork this project
- Create a new branch
- Make your changes
- Merge your branch to master
- Create a pull request and explain your changes


## License

See the [LICENSE](./LICENSE) file for licensing information.
