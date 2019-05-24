/*
 * test_models_PCFNN_graph.c
 *
 */


#include <criterion/criterion.h>
#include <criterion/redirect.h>

#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"
#include "ANN/models/PCFNN/feedforward.h"
#include "ANN/tools.h"
#include "ANN/models/PCFNN/graph.h"


Test(PCFNN_GRAPH, CreateGraph)
{
    struct PCFNN_NETWORK *net = PCFNN_NETWORK_new();
    struct PCFNN_LAYER *l1 = PCFNN_LAYER_new_input(32, f_act_input, f_act_input_de);
    struct PCFNN_LAYER *l2 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l3 = PCFNN_LAYER_new(NULL, NULL, NULL);
    struct PCFNN_LAYER *l4 = PCFNN_LAYER_new(NULL, NULL, NULL);
    PCFNN_NETWORK_addl(net, l1);
    PCFNN_NETWORK_addl(net, l2);
    PCFNN_NETWORK_addl(net, l3);
    PCFNN_NETWORK_addl(net, l4);

    PCFNN_LAYER_connect(l1, l2, 32, 6, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l1, l3, 32, 12, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);

    PCFNN_LAYER_connect(l2, l4, 6, 3, 0, 0, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    PCFNN_LAYER_connect(l3, l4, 12, 5, 0, 2, f_init_rand_norm, f_act_sigmoid, f_act_sigmoid_de);
    
    PCFNN_NETWORK_build(net);
    

    char filename[L_tmpnam];
    tmpnam(filename);
    PCFNN_GRAPH_create_graph_to_file(net, filename, "PCFNN_NETWORK");
    //PCFNN_GRAPH_create_graph_to_file(net, "graph.dot", "PCFNN_NETWORK");
    
    FILE *f = fopen(filename, "r");
    cr_expect_file_contents_neq_str(f, "");
    fclose(f);

    PCFNN_NETWORK_free(net);
    remove(filename);
}
