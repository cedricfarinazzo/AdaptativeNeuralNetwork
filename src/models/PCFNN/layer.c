#include <stdlib.h>
#include "ANN/models/PCFNN/neuron.h"
#include "ANN/tools.h"

#include "ANN/models/PCFNN/layer.h"


struct PCFNN_LAYER *PCFNN_LAYER_new(double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double))
{
    struct PCFNN_LAYER *l = malloc(sizeof(struct PCFNN_LAYER));
    if (l == NULL) return NULL;
    l->size = l->nblinks = l->index = 0;
    l->neurons = malloc(sizeof(struct PCFNN_NEURON*) * l->size);
    if (l->neurons == NULL)
    { free(l); return NULL; }
    l->links = malloc(sizeof(struct PCFNN_LAYER_link*) * l->nblinks);
    if (l->links == NULL)
    { free(l->neurons); free(l); return NULL; }
    l->f_init = f_init;
    l->f_act = f_act;
    l->f_act_de = f_act_de;
    return l;
}


void PCFNN_LAYER_clear(struct PCFNN_LAYER *l)
{
    if (l == NULL) return; 
    for(size_t i = 0; i < l->size; ++i)
        PCFNN_NEURON_clear(l->neurons[i]);
}


void PCFNN_LAYER_free(struct PCFNN_LAYER *l)
{
    if (l != NULL)
    {
        for(size_t i = 0; i < l->size; ++i)
            PCFNN_NEURON_free(l->neurons[i]);
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


int PCFNN_LAYER_addn(struct PCFNN_LAYER *l, size_t size, size_t inputs, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double))
{
    if (size <= 0 && inputs == 0) return 1;
    if (f_init == NULL && l->f_init != NULL)
        f_init = l->f_init;
    if (f_init == NULL) return 1;
    if (f_act == NULL && l->f_act != NULL)
        f_act = l->f_act;
    if (f_act == NULL) return 1;
    if (f_act_de == NULL && l->f_act_de != NULL)
        f_act_de = l->f_act_de;
    if (f_act_de == NULL) return 1;
    
    l->size += size;
    l->neurons = realloc(l->neurons, sizeof(struct PCFNN_NEURON*) * l->size);
    if (l->neurons == NULL) return -1;
    for(size_t i = l->size - size; i < l->size; ++i)
    {
        l->neurons[i] = PCFNN_NEURON_new(inputs, f_init, f_act, f_act_de);
        if (l->neurons[i] == NULL) return -1;
    }
    return 0;
}


struct PCFNN_LAYER *PCFNN_LAYER_new_input(size_t size, double(*f_act)(double), double(*f_act_de)(double))
{
    struct PCFNN_LAYER *l = PCFNN_LAYER_new(f_init_input, f_act, f_act_de);
    if (l == NULL) return NULL;
    if (PCFNN_LAYER_addn(l, size, 0, f_init_input, f_act, f_act_de) != 0)
    {   PCFNN_LAYER_free(l); return NULL; }
    l->type = PCFNN_LAYER_INPUT;
    return l;
}


int PCFNN_LAYER_connect(struct PCFNN_LAYER *from, struct PCFNN_LAYER *to,
                       size_t size_from, size_t size_to,
                       size_t offset_from, size_t offset_to,
                       double(*f_init_to)(), double(*f_act_to)(double), double(*f_act_de_to)(double))
{
    if (from == NULL || to == NULL) return 1;
    size_t ifrom = from->nblinks;
    size_t ito = to->nblinks;
    
    struct PCFNN_LAYER_LINK *link = malloc(sizeof(struct PCFNN_LAYER_LINK));

    if (link == NULL) return -1;
    
    ++from->nblinks; ++to->nblinks;
    from->links = realloc(from->links, sizeof(struct PCFNN_LAYER_LINK*) * from->nblinks);
    if (from->links == NULL) { free(link); return -1; }
    to->links = realloc(to->links, sizeof(struct PCFNN_LAYER_LINK*) * to->nblinks);
    if (to->links == NULL) { free(link); return -1; }
    
    link->from = from;
    link->to = to;
    link->index_from = ifrom;
    link->index_to = ito;
    
    link->size_from = size_from;
    link->size_to = size_to;
    
    link->f_init_to = f_init_to;
    link->f_act_to = f_act_to;
    link->f_act_de_to = f_act_de_to;
    
    link->in_from = offset_from;
    link->in_to = offset_to;
    link->isInitFrom = link->isInitTo = 0;
    
    from->links[ifrom] = to->links[ito] = link;
    
    return 0;
}


int PCFNN_LAYER_build(struct PCFNN_LAYER *l)
{
    size_t nbfromlinks = 0; size_t nbtolinks = 0;
    if (l == NULL) return 1;
    for(size_t k = 0; k < l->nblinks; ++k)
    {
        struct PCFNN_LAYER_LINK *link = l->links[k];
        if (link == NULL) continue;
        if (link->from == l && link->isInitFrom == 0) {
            size_t t = link->in_from + link->size_from;
            if (t > l->size) return -1;
            link->isInitFrom = 1; ++nbfromlinks;
        }
        if (link->to == l && link->isInitTo == 0) {
            if (link->in_to > l->size) return 1;
            for(size_t i = link->in_to; i < l->size; ++i) {
                PCFNN_NEURON_addinputs(l->neurons[i], 
                    link->size_from > l->neurons[i]->size ? link->size_from - l->neurons[i]->size : 0);
            }
            if (PCFNN_LAYER_addn(l,
                    link->size_to + link->in_to > l->size ? link->size_to + link->in_to - l->size : 0
                    , link->size_from, link->f_init_to, link->f_act_to, link->f_act_de_to))
                return -1;
            link->isInitTo = 1; ++nbtolinks;
        }
    }
    for(size_t i = 0; i < l->size; ++i)
        PCFNN_NEURON_build(l->neurons[i]);
    if (nbfromlinks == 0)
        l->type = PCFNN_LAYER_OUTPUT;
    else if (nbtolinks == 0)
        l->type = PCFNN_LAYER_INPUT;
    else
        l->type = PCFNN_LAYER_HIDDEN;
    return 0;
}


size_t PCFNN_LAYER_get_ram_usage(struct PCFNN_LAYER *l)
{
    if (l == NULL) return 0;
    size_t usage = sizeof(struct PCFNN_LAYER);
    usage += sizeof(struct PCFNN_NEURON*) * l->size;
    for (size_t i = 0; i < l->size; ++i)
        usage += PCFNN_NEURON_get_ram_usage(l->neurons[i]);
    usage += sizeof(struct PCFNN_LAYER_LINK*) * l->nblinks;
    usage += sizeof(struct PCFNN_LAYER_LINK) * l->nblinks;
    return usage;
}
