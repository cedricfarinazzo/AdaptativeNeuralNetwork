#include <stdlib.h>

#include "neuron.h"


struct CNN_NEURON *CNN_NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double))
{
    if (size == 0 || f_init == NULL)
        return NULL;
    struct CNN_NEURON *n = malloc(sizeof(struct CNN_NEURON));
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
    return n;
}


void CNN_NEURON_clear(struct CNN_NEURON *n)
{
    if (n != NULL)
        n->output = n->activation = n->delta = 0;
}


void CNN_NEURON_free(struct CNN_NEURON *n)
{
    if (n != NULL)
    {
        free(n->weights);
        free(n);
    }
}


double CNN_NEURON_feedforward(struct CNN_NEURON *n, double *inputs, double(*f_act)(double))
{
    if (n == NULL || inputs == NULL)
        return 0;
    if (f_act != NULL)
        n->f_act = f_act;
    if (n->f_act == NULL)
        return 0;
    n->activation = n->bias;
    for (size_t i = 0; i < n->size; ++i)
        n->activation += n->weights[i] * inputs[i];
    n->output = n->f_act(n->activation);
    return n->output;
}


struct CNN_NEURON *CNN_NEURON_clone_stat(struct CNN_NEURON *n)
{
    if (n == NULL) return NULL;
    return CNN_NEURON_new(n->size, n->f_init, n->f_act);
}


struct CNN_NEURON *CNN_NEURON_clone_all(struct CNN_NEURON *n)
{
    if (n == NULL) return NULL;
    struct CNN_NEURON *b = malloc(sizeof(struct CNN_NEURON));
    if (b == NULL)
        return NULL;
    b->weights = malloc(sizeof(double) * n->size);
    if (b->weights == NULL)
    { free(b); return NULL; }
    b->bias = n->bias; b->output = n->output; b->activation = n->activation;
    b->delta = n->delta; b->f_init = n->f_init; b->f_act = n->f_act; b->size = n->size;
    for(size_t i = 0; i < b->size; ++i)
        b->weights[i] = n->weights[i];
    return b;
}
