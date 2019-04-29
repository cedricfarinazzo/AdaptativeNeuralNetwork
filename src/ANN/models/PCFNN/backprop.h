#ifndef _ANN_MODELS_PCFNN_BACKPROP_H
#define _ANN_MODELS_PCFNN_BACKPROP_H

#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"

void PCFNN_NETWORK_backprop(struct PCFNN_NETWORK *net, double *target, double eta, double(*f_cost)(double, double));

#endif /* _ANN_MODELS_PCFNN_BACKPROP_H */
