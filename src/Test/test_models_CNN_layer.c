/*
 * test_models_CNN_layer.c
 *
 */


#include <criterion/criterion.h>

#include "../ANN/models/CNN/neuron.h"
#include "../ANN/models/CNN/layer.h"
#include "../ANN/tools.h"


Test(CNN_LAYER, Init)
{
    struct CNN_LAYER *l = CNN_LAYER_new(NULL, NULL);
    cr_expect_not_null(l);
    cr_expect_eq(l->size, 0);
    cr_expect_eq(l->nblinks, 0);

    CNN_LAYER_free(l);
}

Test(CNN_LAYER, InitInput)
{
    struct CNN_LAYER *l = CNN_LAYER_new_input(42, f_init_input, f_act_input);
    cr_expect_eq(l->size, 42);
    cr_expect_eq(l->nblinks, 0);
    
    cr_assert_not_null(l->neurons);
    for(size_t i = 0; i < 42; ++i)
    {
        cr_assert_not_null(l->neurons[i]);
        cr_expect_eq(l->neurons[i]->size, 1);
    }
    
    CNN_LAYER_free(l);
}

Test(CNN_LAYER, Connect)
{
    struct CNN_LAYER *l1 = CNN_LAYER_new_input(10, f_init_input, f_act_input);
    struct CNN_LAYER *l2 = CNN_LAYER_new(NULL, NULL);
    
    CNN_LAYER_connect(l1, l2, 10, 25, 0, 0, f_init_rand_norm, f_act_sigmoid);
    
    CNN_LAYER_free(l1);
    CNN_LAYER_free(l2);
}
