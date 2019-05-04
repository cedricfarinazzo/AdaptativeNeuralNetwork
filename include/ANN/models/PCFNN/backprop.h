#ifndef _ANN_MODELS_PCFNN_BACKPROP_H
#define _ANN_MODELS_PCFNN_BACKPROP_H

#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


void PCFNN_NETWORK_backprop(struct PCFNN_NETWORK *net, double *target, double eta, double alpha, double(*f_cost)(double, double));


void PCFNN_NETWORK_apply_delta(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_BACKPROP_H */
