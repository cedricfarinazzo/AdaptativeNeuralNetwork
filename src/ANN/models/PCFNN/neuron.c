#include <stdlib.h>

#include "neuron.h"


struct PCFNN_NEURON *PCFNN_NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double))
{
    if (f_init == NULL)
        return NULL;
    struct PCFNN_NEURON *n = malloc(sizeof(struct PCFNN_NEURON));
    if (n == NULL)
        return NULL;
    n->weights = malloc(sizeof(double) * size);
    if (n->weights == NULL)
    { free(n); return NULL; }
    n->size = size; n->output = n->activation = n->delta = 0;
    n->bias = f_init();
    for (size_t i = 0; i < size; ++i)
        n->weights[i] = f_init();
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
        free(n->weights);
        free(n);
    }
}


void PCFNN_NEURON_addinputs(struct PCFNN_NEURON *n, size_t inputs)
{
    if (n == NULL || inputs <= 0) return;
    size_t i = n->size;
    n->size += inputs;
    n->weights = realloc(n->weights, sizeof(double) * n->size);
    for (; i < n->size; ++i)
        n->weights[i] = n->f_init();
}


double PCFNN_NEURON_feedforward(struct PCFNN_NEURON *n, double *inputs, double(*f_act)(double), double(*f_act_de)(double))
{
    if (n == NULL || inputs == NULL)
        return 0;
    if (f_act != NULL)
        n->f_act = f_act;
    if (n->f_act == NULL)
        return 0;
    if (f_act_de != NULL)
        n->f_act_de = f_act_de;
    if (n->f_act_de == NULL)
        return 0;
    n->activation = n->bias;
    for (size_t i = 0; i < n->size; ++i)
        n->activation += n->weights[i] * inputs[i];
    n->output = n->f_act(n->activation);
    return n->output;
}


struct PCFNN_NEURON *PCFNN_NEURON_clone_stat(struct PCFNN_NEURON *n)
{
    if (n == NULL) return NULL;
    return PCFNN_NEURON_new(n->size, n->f_init, n->f_act, n->f_act_de);
}


struct PCFNN_NEURON *PCFNN_NEURON_clone_all(struct PCFNN_NEURON *n)
{
    if (n == NULL) return NULL;
    struct PCFNN_NEURON *b = malloc(sizeof(struct PCFNN_NEURON));
    if (b == NULL)
        return NULL;
    b->weights = malloc(sizeof(double) * n->size);
    if (b->weights == NULL)
    { free(b); return NULL; }
    b->bias = n->bias; b->output = n->output; b->activation = n->activation;
    b->delta = n->delta; b->f_init = n->f_init; b->f_act = n->f_act; b->f_act_de = n->f_act_de; b->size = n->size;
    for(size_t i = 0; i < b->size; ++i)
        b->weights[i] = n->weights[i];
    return b;
}
