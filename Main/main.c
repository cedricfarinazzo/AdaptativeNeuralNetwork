#include <stdlib.h>
#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"
#include "ANN/models/PCFNN/feedforward.h"
#include "ANN/models/PCFNN/batch.h"
#include "ANN/models/PCFNN/backprop.h"
#include "ANN/tools.h"
#include "ANN/models/PCFNN/train.h"

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
    srand(time(NULL));

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
                         4, 0.0, NULL, 1, 2, 50000, 0.6, f_cost_quadratic_loss_de);

    for(size_t j = 0; j < 4; ++j) 
    {
        PCFNN_NETWORK_feedforward(net, inputs[j]);
        double *out = PCFNN_NETWORK_get_output(net);
        printf("\n %f XOR %f = %f | expected: %f", inputs[j][0], inputs[j][1], out[0], target[j][0]);
        free(out);
    }
    printf("\n");
    PCFNN_NETWORK_free(net);

    return EXIT_SUCCESS;
}
