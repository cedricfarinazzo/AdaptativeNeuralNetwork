#include <stdlib.h>
#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"

#include "ANN/models/PCFNN/batch.h"


void PCFNN_NETWORK_init_batch(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
        {
            net->layers[l]->neurons[n]->wdelta = calloc(net->layers[l]->neurons[n]->size, sizeof(double));
            net->layers[l]->neurons[n]->lastdw = calloc(net->layers[l]->neurons[n]->size, sizeof(double));
            net->layers[l]->neurons[n]->bdelta = 0;
            net->layers[l]->neurons[n]->dsum = 0;
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
                free(net->layers[l]->neurons[n]->lastdw);
                net->layers[l]->neurons[n]->lastdw = NULL;
                net->layers[l]->neurons[n]->bdelta = 0;
            }
}

void PCFNN_NETWORK_clear_batch(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
        {
            for(size_t i = 0; i < net->layers[l]->neurons[n]->size; ++i)
                if (net->layers[l]->neurons[n]->wdelta != NULL)
                    net->layers[l]->neurons[n]->wdelta[i] = 0;
            net->layers[l]->neurons[n]->bdelta = 0;
            net->layers[l]->neurons[n]->dsum = 0;
        }
}

void PCFNN_NETWORK_clear_batch_all(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
        {
            for(size_t i = 0; i < net->layers[l]->neurons[n]->size; ++i)
            {
                if (net->layers[l]->neurons[n]->wdelta != NULL)
                    net->layers[l]->neurons[n]->wdelta[i] = 0;
                if (net->layers[l]->neurons[n]->lastdw != NULL)
                    net->layers[l]->neurons[n]->lastdw[i] = 0;
            }
            net->layers[l]->neurons[n]->bdelta = 0;
        }
}
