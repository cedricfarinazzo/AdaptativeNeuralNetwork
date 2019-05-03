#ifndef _ANN_MODELS_PCFNN_TRAIN_H
#define _ANN_MODELS_PCFNN_TRAIN_H

#include <stdlib.h>
#include <math.h> 
#include "neuron.h"
#include "layer.h"
#include "network.h"
#include "feedforward.h"
#include "backprop.h"
#include "batch.h"


int PCFNN_NETWORK_train(struct PCFNN_NETWORK *net, double **data, double **target,
                         size_t size, double validation_split, int(*f_val)(double, double),
                         int shuffle, unsigned long batch_size, size_t epochs, double eta, 
                         double(*f_cost)(double, double), double *status);

#endif /* _ANN_MODELS_PCFNN_TRAIN_H */
