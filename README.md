# AdaptativeNeuralNetwork ANN (0.1.1.5)

![ANN logo](./.images/logo.medium.png "ANN logo")

### Build status And Code coverage report

| Branch | Build Status | Code Coverage |
|--------|--------------|---------------|
|        |              |               |
| Master |  [![pipeline status](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/master/pipeline.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/master)  |  [![coverage report](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/master/coverage.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/master)  |
|        |              |               |
| Develop|  [![pipeline status](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/develop/pipeline.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/develop)  |  [![coverage report](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/badges/develop/coverage.svg)](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork/commits/develop)  |


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

### PyANN

A python API for ANN

- build from source the c library
- run the following command
```
sudo make pyann_install
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
