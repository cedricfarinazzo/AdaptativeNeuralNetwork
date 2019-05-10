/*
 * test_models_PCFNN_file.c
 *
 */

#include <stdio.h>

#include <criterion/criterion.h>

#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"
#include "ANN/models/PCFNN/file.h"


void __PCFNN_NETWORK_cmp(struct PCFNN_NETWORK *a, struct PCFNN_NETWORK *b)
{
    cr_expect_eq(a->size, a->size);
    
    for(size_t l = 0; l < a->size; ++l)
        for(size_t n = 0; n < a->layers[l]->size; ++n)
        {
            cr_expect_eq(a->layers[l]->neurons[n]->bias, b->layers[l]->neurons[n]->bias);
            for(size_t w = 0; w < a->layers[l]->neurons[n]->size; ++w)
                cr_expect_eq(a->layers[l]->neurons[n]->weights[w], b->layers[l]->neurons[n]->weights[w]);
        }
}


Test(PCFNN_FILE, Save)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l11 = PCFNN_LAYER_new_input(784, f_act_input, f_act_input_de); PCFNN_NETWORK_addl(net, l11);
    struct PCFNN_LAYER *l12 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(net, l12);
    struct PCFNN_LAYER *l13 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(net, l13);
    struct PCFNN_LAYER *l14 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(net, l14);

    PCFNN_LAYER_connect(l11, l12, 784, 250, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l11, l13, 784, 350, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_LAYER_connect(l12, l14, 250, 10, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l13, l14, 350, 42, 0, 9, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    FILE *f = tmpfile();
    cr_expect_eq(PCFNN_NETWORK_save_conf(net, f), 1);
    
    PCFNN_NETWORK_free(net);
    fclose(f);
}


Test(PCFNN_FILE, SaveRestore)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l11 = PCFNN_LAYER_new_input(784, f_act_input, f_act_input_de); PCFNN_NETWORK_addl(net, l11);
    struct PCFNN_LAYER *l12 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(net, l12);
    struct PCFNN_LAYER *l13 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(net, l13);
    struct PCFNN_LAYER *l14 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(net, l14);

    PCFNN_LAYER_connect(l11, l12, 784, 250, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l11, l13, 784, 350, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_LAYER_connect(l12, l14, 250, 10, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l13, l14, 350, 42, 0, 9, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(net);

    char filename[L_tmpnam];
    tmpnam(filename);

    FILE *f = fopen(filename, "w+");
    cr_assert_eq(PCFNN_NETWORK_save_conf(net, f), 1);
    fseek(f, 0, SEEK_SET);

    struct PCFNN_NETWORK *new = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l21 = PCFNN_LAYER_new_input(784, f_act_input, f_act_input_de); PCFNN_NETWORK_addl(new, l21);
    struct PCFNN_LAYER *l22 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(new, l22);
    struct PCFNN_LAYER *l23 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(new, l23);
    struct PCFNN_LAYER *l24 = PCFNN_LAYER_new(NULL, NULL, NULL); PCFNN_NETWORK_addl(new, l24);
    
    PCFNN_LAYER_connect(l21, l22, 784, 250, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l21, l23, 784, 350, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_LAYER_connect(l22, l24, 250, 10, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l23, l24, 350, 42, 0, 9, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_NETWORK_build(new);

    cr_assert_eq(PCFNN_NETWORK_load_conf(new, f), 1);

    __PCFNN_NETWORK_cmp(new, net);

    PCFNN_NETWORK_free(net);
    PCFNN_NETWORK_free(new);
    fclose(f);
}
