#include <stdlib.h>
#include "layer.h"

#include "network.h"


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
    if (net == NULL) return 1;
    ++net->size;
    net->layers = realloc(net->layers, sizeof(struct PCFNN_LAYER*) * net->size);
    if (net->layers == NULL) return -1;
    net->layers[net->size - 1] = l;
    return 0;
}


void PCFNN_NETWORK_build(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
        for(size_t i = 0; i < net->size; ++i)
            PCFNN_LAYER_build(net->layers[i]);
    for(size_t i = 0; i < net->size; ++i)
    {
        if (net->layers[i]->type == PCFNN_LAYER_INPUT)
        { net->inputl = net->layers[i]; break; }
    }
    for(size_t i = 0; i < net->size; ++i)
    {
        if (net->layers[i]->type == PCFNN_LAYER_OUTPUT)
        { net->outputl = net->layers[i]; break; }
    }
}


void PCFNN_NETWORK_feedforward(struct PCFNN_NETWORK *net, double *inputs)
{
    PCFNN_LAYER_feedforward_input(net->inputl, inputs);
    for(size_t i = 0; i < net->size; ++i)
    {
        if (net->layers[i]->type != PCFNN_LAYER_INPUT)
            PCFNN_LAYER_feedforward(net->layers[i]);
    }
}


double *PCFNN_NETWORK_get_output(struct PCFNN_NETWORK *net)
{
    double *output = malloc(sizeof(double) * net->outputl->size);
    for(size_t i = 0; i < net->outputl->size; ++i)
        output[i] = net->outputl->neurons[i]->output;
    return output;
}

