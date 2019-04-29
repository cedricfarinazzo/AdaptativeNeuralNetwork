/*
 * test_models_PCFNN_backprop.c
 *
 */

#include <stdio.h>

#include <criterion/criterion.h>

#include "../ANN/models/PCFNN/neuron.h"
#include "../ANN/models/PCFNN/layer.h"
#include "../ANN/models/PCFNN/network.h"
#include "../ANN/models/PCFNN/backprop.h"
#include "../ANN/tools.h"


Test(PCFNN_BACKPROP, XorSimple)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); PCFNN_NETWORK_addl(net, l3);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    double input[] = {0, 1};
    double target[] = {1};

    PCFNN_NETWORK_feedforward(net, input);
    double *out1 = PCFNN_NETWORK_get_output(net);
    double before = out1[0];
    free(out1);

    PCFNN_NETWORK_backprop(net, target, 0.1, f_cost_quadratic_loss_de);

    PCFNN_NETWORK_feedforward(net, input);
    double *out2 = PCFNN_NETWORK_get_output(net);
    double after = out2[0];
    free(out2);
  
    cr_expect_gt(after, before);
    
    PCFNN_NETWORK_free(net);
}

Test(PCFNN_BACKPROP, XorSimple2)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); PCFNN_NETWORK_addl(net, l3);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    double input[] = {0, 1};
    double target[] = {1};

    for(size_t i = 0; i < 500; ++i)
    {
        PCFNN_NETWORK_feedforward(net, input);
        PCFNN_NETWORK_backprop(net, target, 0.1, f_cost_quadratic_loss_de);
    }

    PCFNN_NETWORK_feedforward(net, input);
    double *out = PCFNN_NETWORK_get_output(net);
    double v = out[0];
    free(out);
  
    cr_expect_gt(v, 0.75);
    
    PCFNN_NETWORK_free(net);
}

Test(PCFNN_BACKPROP, XorTrain)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); PCFNN_NETWORK_addl(net, l3);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    double i1[] = {0, 0};
    double i2[] = {1, 0};
    double i3[] = {0, 1};
    double i4[] = {1, 1};
    double *inputs[] = {i1, i2, i3, i4};
    double target[] = {0, 1, 1, 0};

    for(size_t i = 0; i < 50000; ++i)
    {
        for(size_t j = 0; j < 4; ++j) 
        {
            PCFNN_NETWORK_feedforward(net, inputs[j]);
            double t[] = {target[j]};
            PCFNN_NETWORK_backprop(net, t, 0.5, f_cost_quadratic_loss_de);
        }
    }

    for(size_t j = 0; j < 4; ++j) 
    {
        PCFNN_NETWORK_feedforward(net, inputs[j]);
        double *out = PCFNN_NETWORK_get_output(net);
        double v = out[0];
        free(out);
        //printf("\n\n\ntarget: %f   |   %f\n\n\n", target[j], v);
        if (target[j] == 1)
            cr_expect_gt(v, 0.75);
        else
            cr_expect_lt(v, 0.25);
    }
    
    PCFNN_NETWORK_free(net);
}

