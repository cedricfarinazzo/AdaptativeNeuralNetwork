#ifndef _ANN_MODELS_CNN_NEURON_H_
#define _ANN_MODELS_CNN_NEURON_H_

#include <stdlib.h>

struct CNN_NEURON {
    size_t size;
    double *weights;
    double bias, output; 
    // INTERNAL
    double activation, delta;
    double(*f_init)();
    double(*f_act)(double);
} CNN_NEURON;


struct CNN_NEURON *CNN_NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double));


void CNN_NEURON_clear(struct CNN_NEURON *n);


void CNN_NEURON_free(struct CNN_NEURON *n);


double CNN_NEURON_feedforward(struct CNN_NEURON *n, double *inputs, double(*f_act)(double));


struct CNN_NEURON *CNN_NEURON_clone_stat(struct CNN_NEURON *n);


struct CNN_NEURON *CNN_NEURON_clone_all(struct CNN_NEURON *n);

#endif /* _ANN_MODELS_CNN_NEURON_H_ */
