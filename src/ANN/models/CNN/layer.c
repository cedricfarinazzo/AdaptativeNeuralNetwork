#include <stdlib.h>
#include "neuron.h"
#include "../../tools.h"

#include "layer.h"


struct CNN_LAYER *CNN_LAYER_new(double(*f_init)(), double(*f_act)(double))
{
    struct CNN_LAYER *l = malloc(sizeof(struct CNN_LAYER));
    if (l == NULL) return NULL;
    l->size = l->nblinks = 0;
    l->neurons = malloc(sizeof(struct CNN_NEURON*) * l->size);
    if (l->neurons == NULL)
    { free(l); return NULL; }
    l->links = malloc(sizeof(struct CNN_LAYER_link*) * l->nblinks);
    if (l->links == NULL)
    { free(l->neurons); free(l); return NULL; }
    l->f_init = f_init;
    l->f_act = f_act;
    return l;
}


void CNN_LAYER_free(struct CNN_LAYER *l)
{
    if (l != NULL)
    {
        for(size_t i = 0; i < l->size; ++i)
            CNN_NEURON_free(l->neurons[i]);
        for(size_t i = 0; i < l->nblinks; ++i)
        {
            if (l->links[i] != NULL)
            {
                if (l->links[i]->from == l)
                    l->links[i]->to->links[l->links[i]->index_to] = NULL;
                else
                    l->links[i]->from->links[l->links[i]->index_from] = NULL;
                free(l->links[i]);
                l->links[i] = NULL;
            }
        }
        free(l->neurons);
        free(l->links);
        free(l);
    }
}


int CNN_LAYER_addn(struct CNN_LAYER *l, size_t size, size_t inputs, double(*f_init)(), double(*f_act)(double))
{
    if (size <= 0 && inputs == 0) return 1;
    if (f_init == NULL && l->f_init != NULL)
        f_init = l->f_init;
    if (f_init == NULL) return 1;
    if (f_act == NULL && l->f_act != NULL)
        f_act = l->f_act;
    if (f_act == NULL) return 1;
    
    l->size += size;
    l->neurons = realloc(l->neurons, sizeof(struct CNN_NEURON*) * l->size);
    if (l->neurons == NULL) return -1;
    for(size_t i = l->size - size; i < l->size; ++i)
    {
        l->neurons[i] = CNN_NEURON_new(inputs, f_init, f_act);
        if (l->neurons[i] == NULL) return -1;
    }
    return 0;
}


struct CNN_LAYER *CNN_LAYER_new_input(size_t size, double(*f_act)(double))
{
    struct CNN_LAYER *l = CNN_LAYER_new(f_init_input, f_act);
    if (l == NULL) return NULL;
    if (CNN_LAYER_addn(l, size, 1, f_init_input, f_act) != 0)
    {   CNN_LAYER_free(l); return NULL; }
    l->type = CNN_LAYER_INPUT;
    return l;
}


int CNN_LAYER_connect(struct CNN_LAYER *from, struct CNN_LAYER *to,
                       size_t size_from, size_t size_to,
                       size_t offset_from, size_t offset_to,
                       double(*f_init_to)(), double(*f_act_to)(double))
{
    if (from == NULL || to == NULL) return 1;
    size_t ifrom = from->nblinks;
    size_t ito = to->nblinks;
    
    struct CNN_LAYER_LINK *link = malloc(sizeof(struct CNN_LAYER_LINK))
    ;
    if (link == NULL) return -1;
    
    ++from->nblinks; ++to->nblinks;
    from->links = realloc(from->links, sizeof(struct CNN_LAYER_LINK*) * from->nblinks);
    if (from->links == NULL) { free(link); return -1; }
    to->links = realloc(to->links, sizeof(struct CNN_LAYER_LINK*) * to->nblinks);
    if (to->links == NULL) { free(link); return -1; }
    
    link->from = from;
    link->to = to;
    link->index_from = ifrom;
    link->index_to = ito;
    
    link->size_from = size_from;
    link->size_to = size_to;
    
    link->f_init_to = f_init_to;
    link->f_act_to = f_act_to;
    
    link->in_from = offset_from;
    link->in_to = offset_to;
    link->isInitFrom = link->isInitTo = 0;
    
    from->links[ifrom] = to->links[ito] = link;
    
    return 0;
}


int CNN_LAYER_build(struct CNN_LAYER *l)
{
    size_t nbfromlinks = 0; size_t nbtolinks = 0;
    if (l == NULL) return 1;
    for(size_t k = 0; k < l->nblinks; ++k)
    {
        struct CNN_LAYER_LINK *link = l->links[k];
        if (link == NULL) continue;
        if (link->from == l && link->isInitFrom == 0) {
            size_t t = link->in_from + link->size_from;
            if (t > l->size) return -1;
            link->isInitFrom = 1; ++nbfromlinks;
        }
        if (link->to == l && link->isInitTo == 0) {
            if (link->in_to > l->size) return 1;
            for(size_t i = link->in_to; i < l->size; ++i) {
                CNN_NEURON_addinputs(l->neurons[i], 
                    link->size_from > l->neurons[i]->size ? link->size_from - l->neurons[i]->size : 0);
            }
            if (CNN_LAYER_addn(l,
                    link->size_to + link->in_to > l->size ? link->size_to + link->in_to - l->size : 0
                    , link->size_from, link->f_init_to, link->f_act_to))
                return -1;
            link->isInitTo = 1; ++nbtolinks;
        }
    }
    if (nbfromlinks == 0)
        l->type = CNN_LAYER_OUTPUT;
    else if (nbtolinks == 0)
        l->type = CNN_LAYER_INPUT;
    else
        l->type = CNN_LAYER_HIDDEN;
    return 0;
}


void CNN_LAYER_feedforward_input(struct CNN_LAYER *l, double *inputs)
{
    if (l == NULL || l->type != CNN_LAYER_INPUT) return;
    for (size_t i = 0; i < l->size; ++i)
    {    
        l->neurons[i]->activation = inputs[i];
        l->neurons[i]->output = l->neurons[i]->f_act(l->neurons[i]->activation);
    }
}


void CNN_LAYER_feedforward(struct CNN_LAYER *l)
{
    if (l == NULL || l->type == CNN_LAYER_INPUT) return;
    
    double **inputs = malloc(sizeof(double*) * l->size);
    for(size_t i = 0; i < l->size; ++i)
        inputs[i] = calloc(l->neurons[i]->size, sizeof(double));
    
    for(size_t k = 0; k < l->nblinks; ++k)
    {
        struct CNN_LAYER_LINK *link = l->links[k];
        if (link == NULL) continue;
        if (link->to == l && link->isInitTo)
        {
            for(size_t i = link->in_to; i < link->size_to; ++i)
            {
                double *inp = inputs[i];
                size_t w = 0;
                while(w < l->neurons[i]->size && inp[w] != 0) ++w;
                for(size_t j = link->in_from; j < link->size_from && w < l->neurons[i]->size; ++j, ++w)
                    inp[w] = link->from->neurons[j]->output;
            }
        }
    } 
    
    for(size_t i = 0; i < l->size; ++i)
        CNN_NEURON_feedforward(l->neurons[i], inputs[i], NULL);
    
    for(size_t i = 0; i < l->size; ++i)
        free(inputs[i]);
    free(inputs);
}


void CNN_LAYER_clear(struct CNN_LAYER *l)
{
    if (l == NULL) return; 
    for(size_t i = 0; i < l->size; ++i)
        CNN_NEURON_clear(l->neurons[i]);
}
