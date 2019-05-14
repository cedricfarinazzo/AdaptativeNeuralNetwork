#include <stdlib.h>
#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"

#include "ANN/models/PCFNN/backprop.h"


void PCFNN_LAYER_propagate(struct PCFNN_LAYER *l)
{
    for(size_t i = 0; i < l->size; ++i)
    {
        for (size_t w = 0; w < l->neurons[i]->size; ++w)
            l->neurons[i]->inputs[w]->dsum += l->neurons[i]->delta * l->neurons[i]->weights[w];
    }
}


void PCFNN_LAYER_backward_hidden(struct PCFNN_LAYER *l)
{
    if (l == NULL) return;
    for(size_t i = 0; i < l->size; ++i)
    {
        l->neurons[i]->delta = l->neurons[i]->f_act_de(l->neurons[i]->activation)
            * l->neurons[i]->dsum;
    }
    PCFNN_LAYER_propagate(l);
}

void PCFNN_LAYER_backward_output(struct PCFNN_LAYER *l, double *target, double(*f_cost)(double, double))
{
    if (l == NULL || target == NULL) return;
    for(size_t i = 0; i < l->size; ++i)
    {
        l->neurons[i]->delta = l->neurons[i]->f_act_de(l->neurons[i]->activation)
            * f_cost(l->neurons[i]->output, target[i]);
    }
    PCFNN_LAYER_propagate(l);
}

void PCFNN_NETWORK_backward(struct PCFNN_NETWORK *net, double *target, double(*f_cost)(double, double))
{
    PCFNN_LAYER_backward_output(net->outputl, target, f_cost);
    for(size_t i = net->size - 1; i > 0; --i)
    {
        if (net->layers[i]->type == PCFNN_LAYER_HIDDEN)
            PCFNN_LAYER_backward_hidden(net->layers[i]);
    }
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
        if (l->neurons[i]->state == PCFNN_NEURON_LOCK) continue;
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

