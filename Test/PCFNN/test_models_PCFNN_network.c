/*
 * test_models_PCFNN_network.c
 *
 */


#include <criterion/criterion.h>

#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"
#include "ANN/models/PCFNN/feedforward.h"
#include "ANN/tools.h"


Test(PCFNN_NETWORK, Init)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    cr_expect_not_null(net);
    cr_expect_eq(net->size, 0);
    cr_expect_null(net->inputl);
    cr_expect_null(net->outputl);

    PCFNN_NETWORK_free(net);
}


Test(PCFNN_NETWORK, AddLayer)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l = PCFNN_LAYER_new_input(42, f_act_input, f_act_input_de);

    cr_expect_eq(PCFNN_NETWORK_addl(net, l), 0);
    cr_expect_not_null(net->layers);
    cr_expect_eq(net->size, 1);
    cr_expect_eq(net->layers[net->size - 1], l);

    PCFNN_NETWORK_free(net);
}


Test(PCFNN_NETWORK, AddLayer2)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l1), 0);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l2), 0);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l3), 0);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    cr_expect_eq(net->size, 3);

    PCFNN_NETWORK_free(net);
}


Test(PCFNN_NETWORK, Build)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l1), 0);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l2), 0);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l3), 0);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    cr_expect_eq(PCFNN_NETWORK_build(net), 0);

    cr_expect_eq(net->size, 3);

    cr_expect_eq(net->inputl, l1);
    cr_expect_eq(net->outputl, l3);
    cr_expect_eq(l2->type, PCFNN_LAYER_HIDDEN);

    PCFNN_NETWORK_free(net);
}


Test(PCFNN_NETWORK, Build2)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(784, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l4 = PCFNN_LAYER_new(NULL, NULL, NULL);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l1), 0);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l2), 0);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l3), 0);
    cr_expect_eq(PCFNN_NETWORK_addl(net, l4), 0);

    cr_expect_eq(PCFNN_LAYER_connect(l1, l2, 784, 250, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    cr_expect_eq(PCFNN_LAYER_connect(l1, l3, 784, 350, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);

    cr_expect_eq(PCFNN_LAYER_connect(l2, l4, 250, 10, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    cr_expect_eq(PCFNN_LAYER_connect(l3, l4, 350, 42, 0, 9, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);

    cr_expect_eq(PCFNN_NETWORK_build(net), 0);

    cr_expect_eq(net->inputl, l1);
    cr_expect_eq(net->outputl, l4);
    cr_expect_eq(l2->type, PCFNN_LAYER_HIDDEN);
    cr_expect_eq(l3->type, PCFNN_LAYER_HIDDEN);

    PCFNN_NETWORK_print_summary(net);
    
    PCFNN_NETWORK_free(net);
}


Test(PCFNN_NETWORK, RamUsage)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();

    cr_expect_neq(PCFNN_NETWORK_get_ram_usage(net), 0);

    PCFNN_NETWORK_free(net);
}


Test(PCFNN_NETWORK, LockState)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); PCFNN_NETWORK_addl(net, l3);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    size_t param[2]; param[0] = param[1] = 0;
    PCFNN_LAYER_summary(l1, param);
    cr_expect_eq(param[0], 2);
    cr_expect_eq(param[1], 0);
    
    PCFNN_LAYER_set_lock_state(l1, PCFNN_NEURON_LOCK, 2, 0);
    param[0] = param[1] = 0;
    PCFNN_LAYER_summary(l1, param);
    cr_expect_eq(param[0], 0);
    cr_expect_eq(param[1], 2);
    
    param[0] = param[1] = 0;
    PCFNN_LAYER_summary(l2, param);
    cr_expect_eq(param[0], 6);
    cr_expect_eq(param[1], 0);
    
    PCFNN_LAYER_set_lock_state(l2, PCFNN_NEURON_LOCK, 1, 1);
    param[0] = param[1] = 0;
    PCFNN_LAYER_summary(l2, param);
    cr_expect_eq(param[0], 3);
    cr_expect_eq(param[1], 3);
    
    param[0] = param[1] = 0;
    PCFNN_LAYER_summary(l3, param);
    cr_expect_eq(param[0], 3);
    cr_expect_eq(param[1], 0);
    
    PCFNN_LAYER_set_lock_state(l3, PCFNN_NEURON_LOCK, 42, 0);
    param[0] = param[1] = 0;
    PCFNN_LAYER_summary(l3, param);
    cr_expect_eq(param[0], 3);
    cr_expect_eq(param[1], 0);

    PCFNN_NETWORK_free(net);
}


Test(PCFNN_NETWORK, BuildFromArray)
{
    size_t xorconf[3] = {2, 2, 1};
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_build_from_array(xorconf, 3, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    
    cr_expect_not_null(net);

    size_t param[5];
    PCFNN_NETWORK_summary(net, param);

    cr_expect_eq(param[0], (2*2) + 2 +(2*1) + 1);
    cr_expect_eq(param[1], 0);
    cr_expect_eq(param[3], 3);
    cr_expect_eq(param[4], 2 + 2 + 1);
    

    PCFNN_NETWORK_free(net);
}
