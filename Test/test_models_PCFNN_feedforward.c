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


Test(PCFNN_FEEDFORWARD, FeedForwardNeuron)
{
    size_t s = 2;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, NULL, NULL);
    PCFNN_NEURON_build(n);

    double inputs[] = {2.5, 3.14159};
    cr_expect_neq(PCFNN_NEURON_feedforward_inputs(n, inputs, f_act_sigmoid, f_act_sigmoid_de), 0);
    PCFNN_NEURON_clear(n);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);

    PCFNN_NEURON_free(n);
}


Test(PCFNN_FEEDFORWARD, FeedForwardNeuron2)
{
    size_t s = 2;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_NEURON_build(n);

    double inputs[] = {2.5, 3.14159};
    cr_expect_neq(PCFNN_NEURON_feedforward_inputs(n, inputs, NULL, NULL), 0);
    PCFNN_NEURON_clear(n);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);

    PCFNN_NEURON_free(n);
}


Test(PCFNN_FEEDFORWARD, FeedForwardLayer)
{
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);

    cr_expect_eq(PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    cr_expect_eq(PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);

    cr_expect_eq(PCFNN_LAYER_build(l1), 0);
    cr_expect_eq(PCFNN_LAYER_build(l2), 0);
    cr_expect_eq(PCFNN_LAYER_build(l3), 0);

    PCFNN_LAYER_clear(l1);
    PCFNN_LAYER_clear(l2);
    PCFNN_LAYER_clear(l3);

    double inputs[] = {1, 0.1};
    PCFNN_LAYER_feedforward_input(l1, inputs);
    PCFNN_LAYER_feedforward(l2);
    PCFNN_LAYER_feedforward(l3);

    cr_expect_neq(l3->neurons[l3->size - 1]->output, 0);

    PCFNN_LAYER_free(l1);
    PCFNN_LAYER_free(l2);
    PCFNN_LAYER_free(l3);
}


Test(PCFNN_FEEDFORWARD, FeedForwardNetwork)
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
    PCFNN_NETWORK_clear(net);

    double inputs[] = {1, 0.1};
    PCFNN_NETWORK_feedforward(net, inputs);
    double *out = PCFNN_NETWORK_get_output(net);
    cr_expect_not_null(out);
    cr_expect_neq(out[0], 0);

    if (out != NULL)
        free(out);

    PCFNN_NETWORK_free(net);
}


Test(PCFNN_FEEDFORWARD, FeedForwardNetwork2)
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

    PCFNN_NETWORK_build(net);

    double inputs[784];
    for (size_t i = 0; i < 784; ++i)
        inputs[i] = f_init_rand_norm();

    PCFNN_NETWORK_feedforward(net, inputs);

    double *out = PCFNN_NETWORK_get_output(net);
    cr_expect_not_null(out);

    for(size_t i = 0; i < net->outputl->size; ++i)
        cr_expect_neq(out[i], 0);

    if (out != NULL)
        free(out);

    PCFNN_NETWORK_free(net);
}


