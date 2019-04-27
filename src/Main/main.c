#include <stdlib.h>
#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include "../ANN/models/PCFNN/neuron.h"
#include "../ANN/models/PCFNN/layer.h"
#include "../ANN/tools.h"


int main(int argc, char *argv[])
{
    srand(time(NULL));

    //Creation of 1 input layer, 2 hidden layers and 1 ouput layer
    struct PCFNN_LAYER *i = PCFNN_LAYER_new_input(2, f_act_input);
    struct PCFNN_LAYER *h1 = PCFNN_LAYER_new(NULL, NULL);
    struct PCFNN_LAYER *h2 = PCFNN_LAYER_new(NULL, NULL);
    struct PCFNN_LAYER *o = PCFNN_LAYER_new(NULL, NULL);

    //Connect input layer to the 2 hidden layers
    PCFNN_LAYER_connect(i, h1, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid);
    PCFNN_LAYER_connect(i, h2, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid);

    //Connect each hidden to the output layer
    PCFNN_LAYER_connect(h1, o, 2, 2, 0, 0, f_init_rand_norm, f_act_sigmoid);
    PCFNN_LAYER_connect(h2, o, 2, 1, 0, 2, f_init_rand_norm, f_act_sigmoid);

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

    return EXIT_SUCCESS;
}
