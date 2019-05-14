#include <stdlib.h>
#include <stdio.h>
#include "ANN/models/PCFNN/layer.h"

#include "ANN/models/PCFNN/network.h"


struct PCFNN_NETWORK *PCFNN_NETWORK_new()
{
    struct PCFNN_NETWORK *net = malloc(sizeof(struct PCFNN_NETWORK));
    if (net == NULL) return NULL;
    net->size = 0;
    net->layers = malloc(sizeof(struct PCFNN_LAYER*) * net->size);
    if (net->layers == NULL) return NULL;
    net->inputl = net->outputl = NULL;
    return net;
}


void PCFNN_NETWORK_free(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t i = 0; i < net->size; ++i)
        PCFNN_LAYER_free(net->layers[i]);
    free(net->layers);
    free(net);
}


void PCFNN_NETWORK_clear(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    for(size_t i = 0; i < net->size; ++i)
        PCFNN_LAYER_clear(net->layers[i]);
}


int PCFNN_NETWORK_addl(struct PCFNN_NETWORK *net, struct PCFNN_LAYER *l)
{
    if (net == NULL || l == NULL) return 1;
    ++net->size;
    net->layers = realloc(net->layers, sizeof(struct PCFNN_LAYER*) * net->size);
    if (net->layers == NULL) return -1;
    net->layers[net->size - 1] = l;
    return 0;
}


int PCFNN_NETWORK_build(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return 1;
    int e = 0;
    for(size_t i = 0; i < net->size && e == 0; ++i)
    {
        e = PCFNN_LAYER_build(net->layers[i]);
        net->layers[i]->index = i;
    }
    for(size_t i = 0; e == 0 && i < net->size; ++i)
    {
        if (net->inputl == NULL && net->layers[i]->type == PCFNN_LAYER_INPUT)
        { net->inputl = net->layers[i]; continue; }
        if (net->outputl == NULL && net->layers[i]->type == PCFNN_LAYER_OUTPUT)
        { net->outputl = net->layers[i]; continue; }
    }
    return e;
}


size_t PCFNN_NETWORK_get_ram_usage(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return 0;
    size_t usage = sizeof(struct PCFNN_NETWORK);
    usage += sizeof(struct PCFNN_LAYER*);
    for (size_t i = 0; i < net->size; ++i)
        usage += PCFNN_LAYER_get_ram_usage(net->layers[i]);
    return usage;
}


void PCFNN_NETWORK_summary(struct PCFNN_NETWORK *net, size_t param[5])
{
    /* 0: number of unlocked parameters
     * 1: number of locked parameters
     * 2: ram usage in bytes
     * 3: number of layer
     * 4: number of neurons
     *
     */
    if (net == NULL || param == NULL) return;
    param[0] = param[1] = param[2] = param[4]= 0;
    param[3] = net->size;
    param[2] = PCFNN_NETWORK_get_ram_usage(net);
    for (size_t i = 0; i < net->size; ++i)
    {
        PCFNN_LAYER_summary(net->layers[i], param);
        param[4] += net->layers[i]->size;
    }
}

void PCFNN_NETWORK_print_summary(struct PCFNN_NETWORK *net)
{
    if (net == NULL) return;
    size_t param[5];
    PCFNN_NETWORK_summary(net, param);
    
    printf("===\n   PCFNN_NETWORK: summary\n* Neural network ram usage: ");
    if (param[2] < 1000)
        printf("%ld o", param[2]);
    else if (param[2] < 1000000 && param[2] > 1000)
        printf("%.2f Ko", (double)param[2] / 1000.);
    else if (param[2] < 1000000000 && param[2] > 1000000)
        printf("%.2f Mo", (double)param[2] / 1000000.);
    else if (param[2] < 1000000000000 && param[2] > 1000000000)
        printf("%.2f Go", (double)param[2] / 1000000000.);
    else
        printf("%.2f To", (double)param[2] / 1000000000000.);
    printf("\n* Number of layers: %ld\n* Number of neurons: %ld\n--\n* Layer summary: \n", param[3], param[4]);
    for (size_t i = 0; i < net->size; ++i)
    {
        char type = 'H';
        if (net->layers[i]->type == PCFNN_LAYER_INPUT)
            type = 'I';
        if (net->layers[i]->type == PCFNN_LAYER_OUTPUT)
            type = 'O';
        printf("    - [%c] layer: n°%ld    : %ld neurons\n"
               "             links: \n", type, i, net->layers[i]->size);
        for (size_t j = 0; j < net->layers[i]->nblinks; ++j)
        {
            struct PCFNN_LAYER_LINK *link = net->layers[i]->links[j];
            printf("                   - %ld -> %ld | (%ld, %ld) -> (%ld, %ld)\n", 
                   link->from->index, link->to->index, link->offset_from, link->size_from, link->offset_to, link->size_to);
        }
    }
    printf("--\n* Number of unlocked parameters: %ld\n* Number of locked parameters: %ld\n===\n", param[0], param[1]);
}
