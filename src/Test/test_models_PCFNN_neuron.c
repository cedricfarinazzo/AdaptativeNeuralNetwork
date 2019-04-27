/*
 * test_models_PCFNN_neuron.c
 *
 */

#include <criterion/criterion.h>

#include "../ANN/models/PCFNN/neuron.h"
#include "../ANN/tools.h"


Test(PCFNN_NEURON, Init)
{
    size_t s = 2;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    cr_expect_not_null(n);
    cr_expect_eq(n->size, s);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);
    
    PCFNN_NEURON_free(n);
}


Test(PCFNN_NEURON, WrongInit)
{
    size_t s = 0;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, NULL, NULL, NULL);
    cr_expect_null(n);
    PCFNN_NEURON_free(n);
}


Test(PCFNN_NEURON, feedforward)
{
    size_t s = 2;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, NULL, NULL);
    
    double inputs[] = {2.5, 3.14159};
    cr_expect_neq(PCFNN_NEURON_feedforward(n, inputs, f_act_sigmoid, f_act_sigmoid_de), 0);
    PCFNN_NEURON_clear(n);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);

    PCFNN_NEURON_free(n);
}


Test(PCFNN_NEURON, feedforward2)
{
    size_t s = 2;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    
    double inputs[] = {2.5, 3.14159};
    cr_expect_neq(PCFNN_NEURON_feedforward(n, inputs, NULL, NULL), 0);
    PCFNN_NEURON_clear(n);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);

    PCFNN_NEURON_free(n);
}


Test(PCFNN_NEURON, Clone_stat)
{
    size_t s = 3;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, NULL, NULL);
    
    struct PCFNN_NEURON *b = PCFNN_NEURON_clone_stat(n);
    cr_expect_not_null(b);
    cr_expect_eq(b->size, n->size);
    cr_expect_eq(b->f_init, n->f_init);
    cr_expect_eq(b->f_act, n->f_act);
    
    cr_expect_neq(b->bias, n->bias);
    
    PCFNN_NEURON_free(n);
    PCFNN_NEURON_free(b);
}


Test(PCFNN_NEURON, Clone_all)
{
    size_t s = 42;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    
    struct PCFNN_NEURON *b = PCFNN_NEURON_clone_all(n);
    cr_expect_not_null(b);
    cr_expect_eq(b->size, n->size);
    cr_expect_eq(b->f_init, n->f_init);
    cr_expect_eq(b->f_act, n->f_act);
    
    cr_expect_eq(b->bias, n->bias);
    cr_expect_eq(b->output, 0);
    cr_expect_eq(b->activation, 0);
    cr_expect_eq(b->delta, 0);
    
    for(size_t i = 0; i < n->size; ++i)
        cr_expect_eq(b->weights[i], n->weights[i]);
    
    PCFNN_NEURON_free(n);
    PCFNN_NEURON_free(b);
}


Test(PCFNN_NEURON, AllocFail)
{
    size_t s = 3141592653589793238;
    struct PCFNN_NEURON *n = PCFNN_NEURON_new(s, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    cr_expect_null(n);
    PCFNN_NEURON_free(n);
}
