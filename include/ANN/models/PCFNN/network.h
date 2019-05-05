#ifndef _ANN_MODELS_PCFNN_NETWORK_H_
#define _ANN_MODELS_PCFNN_NETWORK_H_

#include <stdlib.h>
#include "layer.h"

struct PCFNN_NETWORK {
    size_t size;
    struct PCFNN_LAYER **layers;
    struct PCFNN_LAYER *inputl, *outputl;
};


struct PCFNN_NETWORK *PCFNN_NETWORK_new();


void PCFNN_NETWORK_free(struct PCFNN_NETWORK *net);


void PCFNN_NETWORK_clear(struct PCFNN_NETWORK *net);


int PCFNN_NETWORK_addl(struct PCFNN_NETWORK *net, struct PCFNN_LAYER *l);


void PCFNN_NETWORK_build(struct PCFNN_NETWORK *net);


double *PCFNN_NETWORK_get_output(struct PCFNN_NETWORK *net);


size_t PCFNN_NETWORK_get_ram_usage(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_NETWORK_H_ */
