#include <stdlib.h>
#include "ANN/models/PCFNN/layer.h"

#include "ANN/models/PCFNN/network.h"


struct PCFNN_NETWORK *PCFNN_NETWORK_new()
{
    struct PCFNN_NETWORK *net = malloc(sizeof(struct PCFNN_NETWORK));
    if (net == NULL) return NULL;
    net->size = 0;
    net->layers = malloc(sizeof(struct PCFNN_LAYER*) * net->size);
    if (net->layers == NULL) return NULL;
    net->inputl = net->outputl = NULL;
    return net;
}


void PCFNN_NETWORK_free(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t i = 0; i < net->size; ++i)
        PCFNN_LAYER_free(net->layers[i]);
    free(net->layers);
    free(net);
}


void PCFNN_NETWORK_clear(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t i = 0; i < net->size; ++i)
        PCFNN_LAYER_clear(net->layers[i]);
}


int PCFNN_NETWORK_addl(struct PCFNN_NETWORK *net, struct PCFNN_LAYER *l)
{
    if (net == NULL || l == NULL) return 1;
    ++net->size;
    net->layers = realloc(net->layers, sizeof(struct PCFNN_LAYER*) * net->size);
    if (net->layers == NULL) return -1;
    net->layers[net->size - 1] = l;
    return 0;
}


int PCFNN_NETWORK_build(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    int e = 0;
    for(size_t i = 0; i < net->size && e == 0; ++i)
    { 
        e = PCFNN_LAYER_build(net->layers[i]); 
        net->layers[i]->index = i; 
    }
    for(size_t i = 0; e == 0 && i < net->size; ++i)
    {
        if (net->layers[i]->type == PCFNN_LAYER_INPUT)
        { net->inputl = net->layers[i]; break; }
        if (net->layers[i]->type == PCFNN_LAYER_OUTPUT)
        { net->outputl = net->layers[i]; break; }
    }
    return e;
}


size_t PCFNN_NETWORK_get_ram_usage(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return 0;
    size_t usage = sizeof(struct PCFNN_NETWORK);
    usage += sizeof(struct PCFNN_LAYER*) * net->size;
    for (size_t i = 0; i < net->size; ++i)
        usage += PCFNN_LAYER_get_ram_usage(net->layers[i]);
    return usage;
}
