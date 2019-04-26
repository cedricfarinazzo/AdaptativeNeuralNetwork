#ifndef _ANN_MODELS_CNN_NETWORK_H_
#define _ANN_MODELS_CNN_NETWORK_H_

#include <stdlib.h>
#include "layer.h"

struct CNN_NETWORK {
    size_t size;
    struct CNN_LAYER **layers;
    struct CNN_LAYER *inputl, *outputl;
};


struct CNN_NETWORK *CNN_NETWORK_new();


void CNN_NETWORK_free(struct CNN_NETWORK *net);


void CNN_NETWORK_clear(struct CNN_NETWORK *net);


int CNN_NETWORK_addl(struct CNN_NETWORK *net, struct CNN_LAYER *l);


void CNN_NETWORK_build(struct CNN_NETWORK *net);


void CNN_NETWORK_feedforward(struct CNN_NETWORK *net, double *inputs);


double *CNN_NETWORK_get_output(struct CNN_NETWORK *net);

#endif /* _ANN_MODELS_CNN_NETWORK_H_ */
