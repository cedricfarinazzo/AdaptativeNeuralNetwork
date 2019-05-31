/*
 * test_models_PCFNN_train.c
 *
 */

#include <stdio.h>

#include <criterion/criterion.h>

#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"
#include "ANN/models/PCFNN/feedforward.h"
#include "ANN/models/PCFNN/batch.h"
#include "ANN/models/PCFNN/backprop.h"
#include "ANN/tools.h"
#include "ANN/models/PCFNN/train.h"

Test(PCFNN_TRAIN, TrainXORStochastic)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); PCFNN_NETWORK_addl(net, l3);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    double i1[] = {0, 0}; double t1[] = {0};
    double i2[] = {1, 0}; double t2[] = {1};
    double i3[] = {0, 1}; double t3[] = {1};
    double i4[] = {1, 1}; double t4[] = {0};
    double *inputs[] = {i1, i2, i3, i4};
    double *target[] = {t1, t2, t3, t4};

    PCFNN_NETWORK_train(net, inputs, target,
                         4, 0.0, 1, 1, 50000, 0.6, 0.9, f_cost_quadratic_loss, f_cost_quadratic_loss_de, NULL);

    for(size_t j = 0; j < 4; ++j)
    {
        PCFNN_NETWORK_feedforward(net, inputs[j]);
        double *out = PCFNN_NETWORK_get_output(net);
        if (target[j][0] == 1)
            cr_expect_gt(out[0], 0.75);
        else
            cr_expect_lt(out[0], 0.25);
        free(out);
    }
    PCFNN_NETWORK_free(net);
}

Test(PCFNN_TRAIN, TrainXORMiniBatch)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); PCFNN_NETWORK_addl(net, l3);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    double i1[] = {0, 0}; double t1[] = {0};
    double i2[] = {1, 0}; double t2[] = {1};
    double i3[] = {0, 1}; double t3[] = {1};
    double i4[] = {1, 1}; double t4[] = {0};
    double *inputs[] = {i1, i2, i3, i4};
    double *target[] = {t1, t2, t3, t4};

    double status;
    PCFNN_NETWORK_train(net, inputs, target,
                         4, 0.0, 1, 2, 50000, 0.7, 0.9, NULL, f_cost_quadratic_loss_de, &status);
    cr_expect_eq(status, 100.0);

    for(size_t j = 0; j < 4; ++j)
    {
        PCFNN_NETWORK_feedforward(net, inputs[j]);
        double *out = PCFNN_NETWORK_get_output(net);
        if (target[j][0] == 1)
            cr_expect_gt(out[0], 0.75);
        else
            cr_expect_lt(out[0], 0.25);
        free(out);
    }
    PCFNN_NETWORK_free(net);
}

Test(PCFNN_TRAIN, TrainValidation)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1); PCFNN_NETWORK_addl(net, l2); PCFNN_NETWORK_addl(net, l3);

    PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    double i1[] = {0, 0}; double t1[] = {0};
    double i2[] = {1, 0}; double t2[] = {1};
    double i3[] = {0, 1}; double t3[] = {1};
    double i4[] = {1, 1}; double t4[] = {0};
    double *inputs[] = {i1, i2, i3, i4};
    double *target[] = {t1, t2, t3, t4};

    double status;
    double *outtrain = PCFNN_NETWORK_train(net, inputs, target,
                         4, 0.0, 1, 2, 25000, 0.7, 0.9, NULL, f_cost_quadratic_loss_de, &status);
    cr_expect_eq(status, 100.0);
    cr_expect_null(outtrain);

    double *out = PCFNN_NETWORK_train(net, inputs, target,
                         4, 1, 0, 0, 0, 0, 0, f_cost_quadratic_loss, f_cost_quadratic_loss_de, NULL);
    cr_expect_not_null(out);

    for (size_t i = 0; i < net->outputl->size; ++i)
        cr_expect_neq(out[i], 0);

    free(out);
    PCFNN_NETWORK_free(net);
}
    
Test(PCFNN_TRAIN, TrainXORfromArray)
{
    size_t len_conf = 3;
    size_t conf[] = {2, 2, 1};
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_build_from_array(conf, len_conf, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    double i1[] = {0, 0}; double t1[] = {0};
    double i2[] = {1, 0}; double t2[] = {1};
    double i3[] = {0, 1}; double t3[] = {1};
    double i4[] = {1, 1}; double t4[] = {0};
    double *inputs[] = {i1, i2, i3, i4};
    double *target[] = {t1, t2, t3, t4};

    double status;
    double *outtrain = PCFNN_NETWORK_train(net, inputs, target,
                         4, 0.0, 1, 2, 25000, 0.7, 0.9, NULL, f_cost_quadratic_loss_de, &status);
    cr_expect_eq(status, 100.0);
    cr_expect_null(outtrain);

    double *out = PCFNN_NETWORK_train(net, inputs, target,
                         4, 1, 0, 0, 0, 0, 0, f_cost_quadratic_loss, f_cost_quadratic_loss_de, NULL);
    cr_expect_not_null(out);

    for (size_t i = 0; i < net->outputl->size; ++i)
        cr_expect_neq(out[i], 0);

    free(out);
    PCFNN_NETWORK_free(net);
}
