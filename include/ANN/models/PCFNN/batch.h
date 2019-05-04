#ifndef _ANN_MODELS_PCFNN_BATCH_H
#define _ANN_MODELS_PCFNN_BATCH_H

#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


void PCFNN_NETWORK_init_batch(struct PCFNN_NETWORK *net);


void PCFNN_NETWORK_free_batch(struct PCFNN_NETWORK *net);


void PCFNN_NETWORK_clear_batch(struct PCFNN_NETWORK *net);


void PCFNN_NETWORK_clear_batch_all(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_BATCH_H */
