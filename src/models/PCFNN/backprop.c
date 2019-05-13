#include <stdlib.h>
#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"

#include "ANN/models/PCFNN/backprop.h"

void PCFNN_LAYER_backward_hidden(struct PCFNN_LAYER *l, size_t **mark)
{
    if (l == NULL) return;
    double *sums = calloc(l->size, sizeof(double));
    for(size_t k = 0; k < l->nblinks; ++k)
    {
        struct PCFNN_LAYER_LINK *link = l->links[k];
        if (link == NULL) continue;
        if (link->from == l && link->isInit)
        {
            struct PCFNN_LAYER *to = link->to;
            size_t *markl = mark[to->index];
            for(size_t i = link->offset_to; i < link->size_to + link->offset_to; ++i)
            {
                for(size_t j = link->offset_from; j < link->size_from + link->offset_from; ++j)
                {
                    sums[j] += to->neurons[i]->delta
                        * to->neurons[i]->weights[
                        markl[i]
                        ];
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
    for(size_t i = 0; i < net->size; ++i)
        mark[i] = calloc(net->layers[i]->size, sizeof(size_t));
    for(size_t i = net->size - 1; i > 0; --i)
    {
        if (net->layers[i]->type == PCFNN_LAYER_HIDDEN)
            PCFNN_LAYER_backward_hidden(net->layers[i], mark);
    }
    for(size_t i = 0; i < net->size; ++i)
        free(mark[i]);
    free(mark);
}

void PCFNN_NEURON_update(struct PCFNN_NEURON *n, double eta, double alpha)
{
    if (n == NULL) return;
    for(size_t i = 0; i < n->size; ++i)
    {
        double dw = -eta * n->delta * n->inputs[i]->output + alpha * n->lastdw[i];
        n->lastdw[i] = dw;
        n->wdelta[i] += dw;
    }
    n->bdelta += -eta * n->delta;
}

void PCFNN_LAYER_update(struct PCFNN_LAYER *l, double eta, double alpha)
{
    if (l == NULL || l->type == PCFNN_LAYER_INPUT) return;
    for(size_t i = 0; i < l->size; ++i)
        PCFNN_NEURON_update(l->neurons[i], eta, alpha);
}


void PCFNN_NETWORK_update(struct PCFNN_NETWORK *net, double eta, double alpha)
{
    for(size_t i = net->size - 1; i > 0; --i)
        PCFNN_LAYER_update(net->layers[i], eta, alpha);
}


void PCFNN_NETWORK_backprop(struct PCFNN_NETWORK *net, double *target, double eta, double alpha, double(*f_cost)(double, double))
{
    if (net == NULL || target == NULL || f_cost == NULL) return;
    PCFNN_NETWORK_backward(net, target, f_cost);
    PCFNN_NETWORK_update(net, eta, alpha);
}


void PCFNN_LAYER_apply_delta(struct PCFNN_LAYER *l)
{
    if (l == NULL) return;
    for (size_t i = 0; i < l->size; ++i)
    {
        l->neurons[i]->bias += l->neurons[i]->bdelta;
        for(size_t j = 0; j < l->neurons[i]->size; ++j)
            l->neurons[i]->weights[j] += l->neurons[i]->wdelta[j];
        PCFNN_NEURON_clear(l->neurons[i]);
    }
}


void PCFNN_NETWORK_apply_delta(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t i = net->size - 1; i > 0; --i)
        PCFNN_LAYER_apply_delta(net->layers[i]);
}

