#ifndef _ANN_MODELS_CNN_LAYER_H_
#define _ANN_MODELS_CNN_LAYER_H_

#include <stdlib.h>
#include "neuron.h"

struct CNN_LAYER;

struct CNN_LAYER_link {
    int index_from, index_to;
    struct CNN_LAYER *from, *to;
    size_t size_from, size_to;
    double(*f_init_to)(); 
    double(*f_act_to)(double);
    int isInitFrom, isInitTo;
    size_t in_from, in_to;
} CNN_LAYER_link;

struct CNN_LAYER {
    size_t size;
    struct CNN_NEURON **neurons;
    size_t nblinks;
    struct CNN_LAYER_link **links;
    double(*f_init)();
    double(*f_act)(double);
    int isInput;
} CNN_LAYER;

#endif /* _ANN_MODELS_CNN_LAYER_H_ */
