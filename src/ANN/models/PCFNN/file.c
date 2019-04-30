#include <stdlib.h>
#include <stdio.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"

#include "file.h"


int PCFNN_NETWORK_save_conf(struct PCFNN_NETWORK *net, FILE *fout)
{
    if (net == NULL || fout == NULL) return -1;
    
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
        {
            fwrite(&net->layers[l]->neurons[n]->bias, sizeof(double), 1, fout);
            for(size_t w = 0; w < net->layers[l]->neurons[n]->size; ++w)
                fwrite(&net->layers[l]->neurons[n]->weights[w], sizeof(double), 1, fout);
        }
    
    return 1;
}

int PCFNN_NETWORK_load_conf(struct PCFNN_NETWORK *net, FILE *fin)
{
    if (net == NULL || fin == NULL) return -1;

    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
        {
            fread(&net->layers[l]->neurons[n]->bias, sizeof(double), 1, fin);
            for(size_t w = 0; w < net->layers[l]->neurons[n]->size; ++w)
                fread(&net->layers[l]->neurons[n]->weights[w], sizeof(double), 1, fin);
        }
    
    return 1;
}
