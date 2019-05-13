#include <stdlib.h>

#include "ANN/models/PCFNN/neuron.h"


struct PCFNN_NEURON *PCFNN_NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double))
{
    if (f_init == NULL) return NULL;
    struct PCFNN_NEURON *n = calloc(1, sizeof(struct PCFNN_NEURON));
    if (n == NULL) return NULL;
    n->size = size;
    n->bias = f_init();
    n->f_init = f_init;
    n->f_act = f_act;
    n->f_act_de = f_act_de;
    return n;
}


void PCFNN_NEURON_clear(struct PCFNN_NEURON *n)
{
    if (n != NULL)
        n->output = n->activation = n->delta = 0;
}


void PCFNN_NEURON_free(struct PCFNN_NEURON *n)
{
    if (n != NULL)
    {
        if (n->inputs != NULL)
            free(n->inputs);
        if (n->weights != NULL)
            free(n->weights);
        free(n);
    }
}


void PCFNN_NEURON_addinputs(struct PCFNN_NEURON *n, size_t inputs)
{
    if (n == NULL || inputs <= 0) return;
    n->size += inputs;
}


void PCFNN_NEURON_build(struct PCFNN_NEURON *n)
{
    if (n == NULL) return;
    n->weights = malloc(sizeof(double) * n->size);
    if (n->weights == NULL) return;
    n->inputs = calloc(n->size, sizeof(struct PCFNN_NEURON*));
    if (n->inputs == NULL) { free(n->weights); n->weights = NULL; return; }
    for (size_t i = 0; i < n->size; ++i)
        n->weights[i] = n->f_init();
}


size_t PCFNN_NEURON_get_ram_usage(struct PCFNN_NEURON *n)
{
    if (n == NULL) return 0;
    size_t usage = sizeof(struct PCFNN_NEURON);
    if (n->weights != NULL) usage += sizeof(double) * n->size;
    if (n->inputs != NULL) usage += sizeof(struct PCFNN_NEURON*) * n->size;
    return usage;
}


struct PCFNN_NEURON *PCFNN_NEURON_clone_stat(struct PCFNN_NEURON *n)
{
    if (n == NULL) return NULL;
    return PCFNN_NEURON_new(n->size, n->f_init, n->f_act, n->f_act_de);
}


struct PCFNN_NEURON *PCFNN_NEURON_clone_all(struct PCFNN_NEURON *n)
{
    struct PCFNN_NEURON *b = PCFNN_NEURON_clone_stat(n);
    if (b == NULL) return NULL;
    if (n->weights != NULL && n->inputs != NULL)
    {
        PCFNN_NEURON_build(b);
        for(size_t i = 0; i < b->size; ++i)
        {
            b->weights[i] = n->weights[i];
            b->inputs[i] = n->inputs[i];
        }
    }
    b->bias = n->bias; b->output = n->output; b->activation = n->activation; b->delta = n->delta;
    b->wdelta = b->lastdw = NULL; b->bdelta = 0;
    return b;
}
