/*
 * test_models_PCFNN_batch.c
 *
 */

#include <stdio.h>

#include <criterion/criterion.h>

#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"
#include "ANN/models/PCFNN/batch.h"


Test(PCFNN_BATCH, InitFree)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(784, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l4 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1);
    PCFNN_NETWORK_addl(net, l2);
    PCFNN_NETWORK_addl(net, l3);
    PCFNN_NETWORK_addl(net, l4);

    PCFNN_LAYER_connect(l1, l2, 784, 250, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l1, l3, 784, 350, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_LAYER_connect(l2, l4, 250, 10, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l3, l4, 350, 42, 0, 9, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    PCFNN_NETWORK_init_batch(net);
    for(size_t l = 0; l < net->size; ++l)
        for(size_t n = 0; n < net->layers[l]->size; ++n)
        {
            cr_expect_not_null(net->layers[l]->neurons[n]->wdelta);
            cr_expect_not_null(net->layers[l]->neurons[n]->lastdw);
            cr_expect_eq(net->layers[l]->neurons[n]->bdelta, 0);
        }

    PCFNN_NETWORK_free_batch(net);

    PCFNN_NETWORK_free(net);
}
