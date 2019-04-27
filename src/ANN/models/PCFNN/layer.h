#ifndef _ANN_MODELS_PCFNN_LAYER_H_
#define _ANN_MODELS_PCFNN_LAYER_H_

#include <stdlib.h>
#include "../../tools.h"
#include "neuron.h"

enum PCFNN_LAYER_TYPE {
    PCFNN_LAYER_INPUT,
    PCFNN_LAYER_HIDDEN,
    PCFNN_LAYER_OUTPUT,
};

struct PCFNN_LAYER;

struct PCFNN_LAYER_LINK {
    int index_from, index_to;
    struct PCFNN_LAYER *from, *to;
    size_t size_from, size_to;
    double(*f_init_to)(); 
    double(*f_act_to)(double);
    int isInitFrom, isInitTo;
    size_t in_from, in_to;
};

struct PCFNN_LAYER {
    size_t size;
    struct PCFNN_NEURON **neurons;
    size_t nblinks;
    struct PCFNN_LAYER_LINK **links;
    double(*f_init)();
    double(*f_act)(double);
    enum PCFNN_LAYER_TYPE type;
};


struct PCFNN_LAYER *PCFNN_LAYER_new(double(*f_init)(), double(*f_act)(double));


void PCFNN_LAYER_free(struct PCFNN_LAYER *l);


int PCFNN_LAYER_addn(struct PCFNN_LAYER *l, size_t size, size_t inputs, double(*f_init)(), double(*f_act)(double));


struct PCFNN_LAYER *PCFNN_LAYER_new_input(size_t size, double(*f_act)(double));


int PCFNN_LAYER_connect(struct PCFNN_LAYER *from, struct PCFNN_LAYER *to,
                       size_t size_from, size_t size_to,
                       size_t offset_from, size_t offset_to,
                       double(*f_init_to)(), double(*f_act_to)(double));


int PCFNN_LAYER_build(struct PCFNN_LAYER *l);


void PCFNN_LAYER_feedforward_input(struct PCFNN_LAYER *l, double *inputs);


void PCFNN_LAYER_feedforward(struct PCFNN_LAYER *l);


void PCFNN_LAYER_clear(struct PCFNN_LAYER *l);

#endif /* _ANN_MODELS_PCFNN_LAYER_H_ */
