#ifndef _ANN_MODELS_CNN_LAYER_H_
#define _ANN_MODELS_CNN_LAYER_H_

#include <stdlib.h>
#include "../../tools.h"
#include "neuron.h"

enum CNN_LAYER_TYPE {
    CNN_LAYER_INPUT,
    CNN_LAYER_HIDDEN,
    CNN_LAYER_OUTPUT,
};

struct CNN_LAYER;

struct CNN_LAYER_LINK {
    int index_from, index_to;
    struct CNN_LAYER *from, *to;
    size_t size_from, size_to;
    double(*f_init_to)(); 
    double(*f_act_to)(double);
    int isInitFrom, isInitTo;
    size_t in_from, in_to;
};

struct CNN_LAYER {
    size_t size;
    struct CNN_NEURON **neurons;
    size_t nblinks;
    struct CNN_LAYER_LINK **links;
    double(*f_init)();
    double(*f_act)(double);
    enum CNN_LAYER_TYPE type;
};


struct CNN_LAYER *CNN_LAYER_new(double(*f_init)(), double(*f_act)(double));


void CNN_LAYER_free(struct CNN_LAYER *l);


int CNN_LAYER_addn(struct CNN_LAYER *l, size_t size, size_t inputs, double(*f_init)(), double(*f_act)(double));


struct CNN_LAYER *CNN_LAYER_new_input(size_t size, double(*f_act)(double));


int CNN_LAYER_connect(struct CNN_LAYER *from, struct CNN_LAYER *to,
                       size_t size_from, size_t size_to,
                       size_t offset_from, size_t offset_to,
                       double(*f_init_to)(), double(*f_act_to)(double));


int CNN_LAYER_build(struct CNN_LAYER *l);


void CNN_LAYER_feedforward_input(struct CNN_LAYER *l, double *inputs);


void CNN_LAYER_feedforward(struct CNN_LAYER *l);


void CNN_LAYER_clear(struct CNN_LAYER *l);

#endif /* _ANN_MODELS_CNN_LAYER_H_ */
