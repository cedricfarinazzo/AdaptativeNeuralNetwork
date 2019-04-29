#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"

#include "backprop.h"

void PCFNN_LAYER_backward_hidden(struct PCFNN_LAYER *l, size_t **mark)
{
    if (l == NULL) return;
    double *sums = calloc(l->size, sizeof(double));
    for(size_t k = 0; k < l->nblinks; ++k)
    {
        struct PCFNN_LAYER_LINK *link = l->links[k];
        if (link == NULL) continue;
        if (link->from == l && link->isInitFrom)
        {
            struct PCFNN_LAYER *to = link->to;
            size_t *markl = mark[to->index];
            for(size_t i = link->in_to; i < link->size_to + link->in_to; ++i)
            {
                for(size_t j = link->in_from; j < link->size_from + link->in_from; ++j)
                {
                    sums[j] += to->neurons[i]->delta 
                        * to->neurons[i]->weights[markl[i]];
                    ++markl[i];
                }
            }
        }
    }
    for(size_t i = 0; i < l->size; ++i)
    { 
        l->neurons[i]->delta = l->neurons[i]->f_act_de(l->neurons[i]->activation)
            * sums[i];
    }
    free(sums);
}

void PCFNN_LAYER_backward_output(struct PCFNN_LAYER *l, double *target, double(*f_cost)(double, double))
{
    if (l == NULL || target == NULL) return;
    for(size_t i = 0; i < l->size; ++i)
    { 
        l->neurons[i]->delta = l->neurons[i]->f_act_de(l->neurons[i]->activation)
            * f_cost(l->neurons[i]->output, target[i]);
    }
}

void PCFNN_NETWORK_backward(struct PCFNN_NETWORK *net, double *target, double(*f_cost)(double, double))
{
    PCFNN_LAYER_backward_output(net->outputl, target, f_cost);
    size_t **mark = malloc(sizeof(size_t*) * net->size);
    for(size_t i = 0; i < net->size; --i)
        mark[i] = calloc(net->layers[i]->size, sizeof(size_t)); 
    for(size_t i = net->size - 1; i > 0; --i)
    {
        if (net->layers[i]->type == PCFNN_LAYER_HIDDEN)
            PCFNN_LAYER_backward_hidden(net->layers[i], mark);
    }
    for(size_t i = 0; i < net->size; --i)
        free(mark[i]);
    free(mark);
}

void PCFNN_NEURON_update(struct PCFNN_NEURON *n, double *inputs, double eta)
{
    if (n == NULL || inputs == NULL) return;
    for(size_t i = 0; i < n->size; ++i)
    {
        n->weights[i] -= eta * n->delta * inputs[i];
    }
    n->bias -= eta * n->delta;
}

void PCFNN_LAYER_update(struct PCFNN_LAYER *l, double eta)
{
    if (l == NULL) return;
    double **inputs = malloc(sizeof(double*) * l->size);
    for(size_t i = 0; i < l->size; ++i)
        inputs[i] = calloc(l->neurons[i]->size, sizeof(double));
    
    for(size_t k = 0; k < l->nblinks; ++k)
    {
        struct PCFNN_LAYER_LINK *link = l->links[k];
        if (link == NULL) continue;
        if (link->to == l && link->isInitTo)
        {
            for(size_t i = link->in_to; i < link->size_to + link->in_to; ++i)
            {
                double *inp = inputs[i];
                size_t w = 0;
                while(w < l->neurons[i]->size && inp[w] != 0) ++w;
                for(size_t j = link->in_from; j < link->size_from + link->in_from && w < l->neurons[i]->size; ++j, ++w)
                    inp[w] = link->from->neurons[j]->output;
            }
        }
    } 
    
    for(size_t i = 0; i < l->size; ++i)
    { PCFNN_NEURON_update(l->neurons[i], inputs[i], eta); free(inputs[i]); }
    free(inputs);
}


void PCFNN_NETWORK_update(struct PCFNN_NETWORK *net, double eta)
{
    for(size_t i = net->size - 1; i > 0; --i)
        PCFNN_LAYER_update(net->layers[i], eta);
}

void PCFNN_NETWORK_backprop(struct PCFNN_NETWORK *net, double *target, double eta, double(*f_cost)(double, double))
{
    PCFNN_NETWORK_backward(net, target, f_cost);
    PCFNN_NETWORK_update(net, eta);
    PCFNN_NETWORK_clear(net);
}
