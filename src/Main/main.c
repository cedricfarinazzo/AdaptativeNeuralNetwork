#include <stdlib.h>
#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include "../ANN/models/PCFNN/neuron.h"
#include "../ANN/models/PCFNN/layer.h"
#include "../ANN/models/PCFNN/network.h"
#include "../ANN/models/PCFNN/feedforward.h"
#include "../ANN/models/PCFNN/backprop.h"
#include "../ANN/tools.h"


int main(int argc, char *argv[])
{
    srand(time(NULL));
if (0) {
    //Creation of 1 input layer, 2 hidden layers and 1 ouput layer
    struct PCFNN_LAYER *i = PCFNN_LAYER_new_input(2, f_act_input, f_act_input);
    struct PCFNN_LAYER *h1 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *h2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *o = PCFNN_LAYER_new(NULL, NULL, NULL);

    //Connect input layer to the 2 hidden layers
    PCFNN_LAYER_connect(i, h1, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(i, h2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    //Connect each hidden to the output layer
    PCFNN_LAYER_connect(h1, o, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(h2, o, 2, 1, 0, 2, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    //Build each layer
    PCFNN_LAYER_build(i);
    PCFNN_LAYER_build(h1);
    PCFNN_LAYER_build(h2);
    PCFNN_LAYER_build(o);
    /* NETWORK DIAGRAM
     *
     *             <input>
     *             | n n |
     *             -------
     *             /     \
     *            /       \
     *           /         \
     *  <Hidden1>          <Hidden2>
     *  | n   n |          | n   n |
     *  ---------          ---------
     *      \                /
     *       \              /
     *        \------|-----/
     *        | n  n |    n |
     *        ---------------
     *            <Output>
     */

    //Free memory allocation
    PCFNN_LAYER_free(i);
    PCFNN_LAYER_free(h1);
    PCFNN_LAYER_free(h2);
    PCFNN_LAYER_free(o);
}

if (0) {
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

    double *inputs = malloc(sizeof(double) * 784);
    for (size_t i = 0; i < 784; ++i)
        inputs[i] = f_init_rand_norm();

    PCFNN_NETWORK_feedforward(net, inputs);
    free(inputs);

    double *out = PCFNN_NETWORK_get_output(net);
    if (out == NULL)
        return EXIT_FAILURE;
    for(size_t i = 0; i < net->outputl->size; ++i)
        printf("%f ", out[i]);

    if (out != NULL)
        free(out);
    

    PCFNN_NETWORK_free(net);
}


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

    for(size_t i = 0; i < 500; ++i)
    {
        PCFNN_NETWORK_feedforward(net, input);
        PCFNN_NETWORK_backprop(net, target, 0.1, f_cost_quadratic_loss_de);
    }

    PCFNN_NETWORK_free(net);


    return EXIT_SUCCESS;
}
