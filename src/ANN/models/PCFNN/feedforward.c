#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"

#include "feedforward.h"


double PCFNN_NEURON_feedforward(struct PCFNN_NEURON *n, double *inputs, double(*f_act)(double), double(*f_act_de)(double))
{
    if (n == NULL) return 0;
    if (f_act != NULL) n->f_act = f_act;
    if (n->f_act == NULL) return 0;
    if (f_act_de != NULL) n->f_act_de = f_act_de;
    if (n->f_act_de == NULL) return 0;
    n->activation = n->bias;
    if (inputs == NULL) inputs = n->inputs;
    for (size_t i = 0; i < n->size; ++i)
        n->activation += n->weights[i] * inputs[i];
    n->output = n->f_act(n->activation);
    return n->output;
}


void PCFNN_LAYER_feedforward_input(struct PCFNN_LAYER *l, double *inputs)
{
    if (l == NULL || l->type != PCFNN_LAYER_INPUT) return;
    for (size_t i = 0; i < l->size; ++i)
    {    
        l->neurons[i]->activation = inputs[i];
        l->neurons[i]->output = l->neurons[i]->f_act(l->neurons[i]->activation);
    }
}


void PCFNN_LAYER_feedforward(struct PCFNN_LAYER *l)
{
    if (l == NULL || l->type == PCFNN_LAYER_INPUT) return;
    size_t w[l->size];
    for(size_t i = 0; i < l->size; ++i) w[i] = 0;
    for(size_t k = 0; k < l->nblinks; ++k)
    {
        struct PCFNN_LAYER_LINK *link = l->links[k];
        if (link == NULL) continue;
        if (link->to == l && link->isInitTo)
        {
            for(size_t i = link->in_to; i < link->size_to + link->in_to; ++i)
            {
                for(size_t j = link->in_from; j < link->size_from + link->in_from && w[i] < l->neurons[i]->size; ++j, ++w[i])
                    l->neurons[i]->inputs[w[i]] = link->from->neurons[j]->output;
            }
        }
    } 
    for(size_t i = 0; i < l->size; ++i)
        PCFNN_NEURON_feedforward(l->neurons[i], NULL, NULL, NULL);
}


void PCFNN_NETWORK_feedforward(struct PCFNN_NETWORK *net, double *inputs)
{
    if (net == NULL) return;
    PCFNN_LAYER_feedforward_input(net->inputl, inputs);
    for(size_t i = 0; i < net->size; ++i)
    {
        if (net->layers[i]->type != PCFNN_LAYER_INPUT)
            PCFNN_LAYER_feedforward(net->layers[i]);
    }
}


double *PCFNN_NETWORK_get_output(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return NULL;
    double *output = malloc(sizeof(double) * net->outputl->size);
    if (output == NULL) return NULL;
    for(size_t i = 0; i < net->outputl->size; ++i)
        output[i] = net->outputl->neurons[i]->output;
    return output;
}
