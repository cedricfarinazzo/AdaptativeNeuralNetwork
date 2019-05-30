# AdaptativeNeuralNetwork ANN (0.1.1.3)

![ANN logo](./.images/logo.medium.png "ANN logo")

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
- [graphviz](https://graphviz.gitlab.io)
- [criterion](https://criterion.readthedocs.io/en/master/) (test only)
- doxygen and perl (for documentation generation)


## AdaptativeNeuralNetwork

A static library containing multiple neural network models written in C


## Examples

See some examples [here](https://github.com/cedricfarinazzo/ANNExample)


## Documentation (Doxygen)

View the documentation of all functions [here](https://adaptativeneuralnetwork.ml/)


## Installation

### From the Arch User Repository (AUR)

- with yay
```
yay -S adaptativeneuralnetwork
```

- with pacman
```
git clone https://aur.archlinux.org/adaptativeneuralnetwork.git
cd adaptativeneuralnetwork
makepkg -sci
```


### Install from source

- Clone this repo
- Then run the following command 
```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_PREFIX=/usr \
    ..
sudo make install
```


## Development version

- clone this repo
- checkout on develop branch

- Build
```
mkdir -p build && cd build && cmake .. && make
```

- Run test
```
make check
```

- Coverage report
```
make coverage
```

- Generate documentation
```
make doc
```


## Contribute

- Fork this project
- Create a new branch
- Make your changes
- Merge your branch to develop
- Create a pull request and explain your changes


## License

See the [LICENSE](./LICENSE) file for licensing information.
