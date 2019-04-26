#include <stdlib.h>
#include "layer.h"

#include "network.h"


struct CNN_NETWORK *CNN_NETWORK_new()
{
    struct CNN_NETWORK *net = malloc(sizeof(struct CNN_NETWORK));
    if (net == NULL) return NULL;
    net->size = 0;
    net->layers = malloc(sizeof(struct CNN_LAYER*) * net->size);
    if (net->layers == NULL) return NULL;
    net->inputl = net->outputl = NULL;
    return net;
}


void CNN_NETWORK_free(struct CNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t i = 0; i < net->size; ++i)
        CNN_LAYER_free(net->layers[i]);
    free(net);
}


void CNN_NETWORK_clear(struct CNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t i = 0; i < net->size; ++i)
        CNN_LAYER_clear(net->layers[i]);
}


int CNN_NETWORK_addl(struct CNN_NETWORK *net, struct CNN_LAYER *l)
{
    if (net == NULL) return 1;
    ++net->size;
    net->layers = realloc(net->layers, sizeof(struct CNN_LAYER*) * net->size);
    if (net->layers == NULL) return -1;
    net->layers[net->size - 1] = l;
    return 0;
}


void CNN_NETWORK_build(struct CNN_NETWORK *net)
{
    if (net == NULL) return;
        for(size_t i = 0; i < net->size; ++i)
            CNN_LAYER_build(net->layers[i]);
    for(size_t i = 0; i < net->size; ++i)
    {
        if (net->layers[i]->type == CNN_LAYER_INPUT)
        { net->inputl = net->layers[i]; break; }
    }
    for(size_t i = 0; i < net->size; ++i)
    {
        if (net->layers[i]->type == CNN_LAYER_OUTPUT)
        { net->outputl = net->layers[i]; break; }
    }
}


void CNN_NETWORK_feedforward(struct CNN_NETWORK *net, double *inputs)
{
    CNN_LAYER_feedforward_input(net->inputl, inputs);
    for(size_t i = 0; i < net->size; ++i)
    {
        if (net->layers[i]->type != CNN_LAYER_INPUT)
            CNN_LAYER_feedforward(net->layers[i]);
    }
}


double *CNN_NETWORK_get_output(struct CNN_NETWORK *net)
{
    double *output = malloc(sizeof(double) * net->outputl->size);
    for(size_t i = 0; i < net->outputl->size; ++i)
        output[i] = net->outputl->neurons[i]->output;
    return output;
}

