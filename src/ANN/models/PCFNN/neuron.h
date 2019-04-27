#ifndef _ANN_MODELS_PCFNN_NEURON_H_
#define _ANN_MODELS_PCFNN_NEURON_H_

#include <stdlib.h>

struct PCFNN_NEURON {
    size_t size;
    double *weights;
    double bias, output; 
    // INTERNAL
    double activation, delta;
    double(*f_init)();
    double(*f_act)(double);
    double(*f_act_de)(double);
};


struct PCFNN_NEURON *PCFNN_NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double));


void PCFNN_NEURON_clear(struct PCFNN_NEURON *n);


void PCFNN_NEURON_free(struct PCFNN_NEURON *n);


void PCFNN_NEURON_addinputs(struct PCFNN_NEURON *n, size_t inputs);


double PCFNN_NEURON_feedforward(struct PCFNN_NEURON *n, double *inputs, double(*f_act)(double), double(*f_act_de)(double));


struct PCFNN_NEURON *PCFNN_NEURON_clone_stat(struct PCFNN_NEURON *n);


struct PCFNN_NEURON *PCFNN_NEURON_clone_all(struct PCFNN_NEURON *n);

#endif /* _ANN_MODELS_PCFNN_NEURON_H_ */
