/*
 * test_models_PCFNN_layer.c
 *
 */


#include <criterion/criterion.h>

#include "../ANN/models/PCFNN/neuron.h"
#include "../ANN/models/PCFNN/layer.h"
#include "../ANN/tools.h"


Test(PCFNN_LAYER, Init)
{
    struct PCFNN_LAYER *l = PCFNN_LAYER_new(NULL, NULL, NULL);
    cr_expect_not_null(l);
    cr_expect_eq(l->size, 0);
    cr_expect_eq(l->nblinks, 0);

    PCFNN_LAYER_free(l);
}


Test(PCFNN_LAYER, InitInput)
{
    struct PCFNN_LAYER *l = PCFNN_LAYER_new_input(42, f_act_input, f_act_input_de);
    cr_expect_eq(l->size, 42);
    cr_expect_eq(l->nblinks, 0);
    
    cr_assert_not_null(l->neurons);
    for(size_t i = 0; i < 42; ++i)
    {
        cr_assert_not_null(l->neurons[i]);
        cr_expect_eq(l->neurons[i]->size, 1);
    }
    
    PCFNN_LAYER_free(l);
}


Test(PCFNN_LAYER, Connect)
{
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(10, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    
    cr_expect_eq(PCFNN_LAYER_connect(l1, l2, 10, 25, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    
    PCFNN_LAYER_free(l1);
    PCFNN_LAYER_free(l2);
}


Test(PCFNN_LAYER, Builder)
{
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    
    cr_expect_eq(PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    cr_expect_eq(PCFNN_LAYER_connect(l2, l3, 2, 1, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    
    cr_expect_eq(PCFNN_LAYER_build(l1), 0);
    cr_expect_eq(PCFNN_LAYER_build(l2), 0);
    cr_expect_eq(PCFNN_LAYER_build(l3), 0);
    
    cr_expect_eq(l1->size, 2);
    cr_expect_eq(l2->size, 2);
    cr_expect_eq(l3->size, 1);
    
    PCFNN_LAYER_free(l1);
    PCFNN_LAYER_free(l2);
    PCFNN_LAYER_free(l3);   
}


Test(PCFNN_LAYER, Builder2)
{
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(2, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l4 = PCFNN_LAYER_new(NULL, NULL, NULL);
    
    cr_expect_eq(PCFNN_LAYER_connect(l1, l2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    cr_expect_eq(PCFNN_LAYER_connect(l1, l3, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    
    cr_expect_eq(PCFNN_LAYER_connect(l2, l4, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    cr_expect_eq(PCFNN_LAYER_connect(l3, l4, 2, 1, 0, 2, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de), 0);
    
    cr_expect_eq(PCFNN_LAYER_build(l1), 0);
    cr_expect_eq(PCFNN_LAYER_build(l2), 0);
    cr_expect_eq(PCFNN_LAYER_build(l3), 0);
    cr_expect_eq(PCFNN_LAYER_build(l4), 0);
    
    cr_expect_eq(l1->size, 2);
    cr_expect_eq(l2->size, 2);
    cr_expect_eq(l3->size, 2);
    cr_expect_eq(l4->size, 3);
    
    PCFNN_LAYER_free(l1);
    PCFNN_LAYER_free(l2);
    PCFNN_LAYER_free(l3);   
    PCFNN_LAYER_free(l4);   
}


Test(PCFNN_LAYER, FeedForward)
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
