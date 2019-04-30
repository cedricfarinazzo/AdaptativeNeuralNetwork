#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"

#include "batch.h"


void PCFNN_NETWORK_init_batch(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
        { 
            net->layers[l]->neurons[n]->wdelta = calloc(net->layers[l]->neurons[n]->size, sizeof(double));
            net->layers[l]->neurons[n]->bdelta = 0; 
        }
}


void PCFNN_NETWORK_free_batch(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
            if (net->layers[l]->neurons[n]->wdelta != NULL)
            {
                free(net->layers[l]->neurons[n]->wdelta); 
                net->layers[l]->neurons[n]->wdelta = NULL; 
                net->layers[l]->neurons[n]->bdelta = 0; 
            }
}

void PCFNN_NETWORK_clear_batch(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
            if (net->layers[l]->neurons[n]->wdelta != NULL)
            {
                for(size_t i = 0; i < net->layers[l]->neurons[n]->size; ++i)
                    net->layers[l]->neurons[n]->wdelta[i] = 0; 
                net->layers[l]->neurons[n]->bdelta = 0; 
            }
}
