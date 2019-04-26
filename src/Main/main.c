#include <stdlib.h>
#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include "../ANN/models/CNN/neuron.h"
#include "../ANN/models/CNN/layer.h"
#include "../ANN/tools.h"


int main(int argc, char *argv[])
{
    srand(time(NULL));

    //Creation of 1 input layer, 2 hidden layers and 1 ouput layer
    struct CNN_LAYER *i = CNN_LAYER_new_input(2, f_init_input, f_act_input);
    struct CNN_LAYER *h1 = CNN_LAYER_new(NULL, NULL);
    struct CNN_LAYER *h2 = CNN_LAYER_new(NULL, NULL);
    struct CNN_LAYER *o = CNN_LAYER_new(NULL, NULL);

    //Connect input layer to the 2 hidden layers
    CNN_LAYER_connect(i, h1, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid);
    CNN_LAYER_connect(i, h2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid);

    //Connect each hidden to the output layer
    CNN_LAYER_connect(h1, o, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid);
    CNN_LAYER_connect(h2, o, 2, 1, 0, 2, f_init_rand_norm, f_act_sigmoid);

    //Build each layer
    CNN_LAYER_build(i);
    CNN_LAYER_build(h1);
    CNN_LAYER_build(h2);
    CNN_LAYER_build(o);
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
    CNN_LAYER_free(i);
    CNN_LAYER_free(h1);
    CNN_LAYER_free(h2);
    CNN_LAYER_free(o);

    return EXIT_SUCCESS;
}
