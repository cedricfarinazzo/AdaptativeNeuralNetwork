#ifndef _ANN_CORE_NEURON_H_
#define _ANN_CORE_NEURON_H_

#include <stdlib.h>

struct NEURON {
    size_t size;
    double *weights;
    double bias, output; 
    // INTERNAL
    double activation, delta;
    double(*f_init)();
    double(*f_act)(double);
} NEURON;


struct NEURON *NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double));


void NEURON_clear(struct NEURON *n);


void NEURON_free(struct NEURON *n);


double NEURON_feedforward(struct NEURON *n, double *inputs, double(*f_act)(double));


struct NEURON *NEURON_clone_stat(struct NEURON *n);


struct NEURON *NEURON_clone_all(struct NEURON *n);

#endif /* _ANN_CORE_NEURON_H_ */
