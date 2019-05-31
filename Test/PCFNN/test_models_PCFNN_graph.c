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
#include "ANN/models/PCFNN/graph.h"
#include "ANN/tools.h"


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
    
    Agraph_t *g = PCFNN_GRAPH_create_graph(net, "PCFNN_NETWORK");
    PCFNN_NETWORK_free(net);
    
    cr_assert_not_null(g);

    Agnode_t *v;
    Agedge_t *e;
    size_t cnt = 0;
    for (v = agfstnode(g); v; v = agnxtnode(g,v))
        for (e = agfstout(g,v); e; e = agnxtout(g,e))
            cnt++;
    cr_expect_neq(cnt, 0);

    agclose(g);
}


Test(PCFNN_GRAPH, CreateGraphToDotFile)
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
    cr_expect_eq(PCFNN_GRAPH_create_graph_to_dot_file(net, "PCFNN_NETWORK", filename), 0);
    PCFNN_NETWORK_free(net);
    
    FILE *f = fopen(filename, "r");
    cr_assert_not_null(f);
    cr_expect_file_contents_neq_str(f, "");
    fclose(f);

    remove(filename);
}


Test(PCFNN_GRAPH, RenderGraphToStream)
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
    FILE *f = fopen(filename, "w+");
    cr_expect_eq(PCFNN_GRAPH_render_graph_to_stream(net, "PCFNN_NETWORK", "png", f), 0);
    PCFNN_NETWORK_free(net);
    fclose(f);
    
    f = fopen(filename, "r");
    cr_assert_not_null(f);
    cr_expect_file_contents_neq_str(f, "");
    fclose(f);

    remove(filename);
}


Test(PCFNN_GRAPH, RenderGraphToFile)
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
    cr_expect_eq(PCFNN_GRAPH_render_graph_to_file(net, "PCFNN_NETWORK", "png", filename), 0);
    PCFNN_NETWORK_free(net);
    
    FILE *f = fopen(filename, "r");
    cr_assert_not_null(f);
    cr_expect_file_contents_neq_str(f, "");
    fclose(f);

    remove(filename);
}
