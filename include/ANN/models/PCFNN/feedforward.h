#ifndef _ANN_MODELS_PCFNN_FEEDFORWARD_H
#define _ANN_MODELS_PCFNN_FEEDFORWARD_H

#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


double PCFNN_NEURON_feedforward(struct PCFNN_NEURON *n, double *inputs, double(*f_act)(double), double(*f_act_de)(double));

void PCFNN_LAYER_feedforward_input(struct PCFNN_LAYER *l, double *inputs);


void PCFNN_LAYER_feedforward(struct PCFNN_LAYER *l);


void PCFNN_NETWORK_feedforward(struct PCFNN_NETWORK *net, double *inputs);


double *PCFNN_NETWORK_get_output(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_FEEDFORWARD_H */
