#include <stdlib.h>
#include <stdio.h>
#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"

#include <graphviz/gvc.h>
#ifdef WITH_CGRAPH
#include <graphviz/cgraph.h>
#else
#include <graphviz/graph.h>
#endif

#include "ANN/models/PCFNN/graph.h"


void PCFNN_GRAPH_create_graph_to_file(struct PCFNN_NETWORK *net, char *out, char *graph_name)
{
    if (net == NULL || out == NULL) return;

    Agraph_t *g;
    Agraph_t *layers[net->size];
    Agnode_t **nodes[net->size];

#ifdef WITH_CGRAPH
    g = agopen(graph_name, Agdirected, NULL);
#else
    aginit();
    g = agopen(graph_name, AGDIGRAPH);
#endif
    if (g == NULL) return;

    for (size_t l = 0; l < net->size; ++l)
    {
        char lbuff[42];
        sprintf(lbuff, "L%ld", net->layers[l]->index);
        layers[l] = agsubg(g, lbuff, 1);
        nodes[l] = malloc(sizeof(Agnode_t*) * net->layers[l]->size);
        for (size_t n = 0; n < net->layers[l]->size; ++n)
        {
            char buff[42];
            sprintf(buff, "L%ld_%ld", net->layers[l]->index, n);
#ifdef WITH_CGRAPH
            nodes[l][n] = agnode(layers[l], buff, 1);
#else
            nodes[l][n] = agnode(layers[l], buff);
#endif
        }
    }

    for (size_t l = 0; l < net->size; ++l)
    {
        for (size_t k = 0; k < net->layers[l]->nblinks; ++k)
        {
            struct PCFNN_LAYER_LINK *link = net->layers[l]->links[k];
            if (link != NULL && net->layers[l] == link->from)
            {
                for(size_t n = link->offset_to; n < link->size_to + link->offset_to; ++n)
                {
                    for (size_t m = link->offset_from; m < link->size_from + link->offset_from; ++m)
                    {
#ifdef WITH_CGRAPH
                        agedge(g, nodes[link->from->index][m], nodes[link->to->index][n], NULL, 1);
#else
                        agedge(g, nodes[link->from->index][m], nodes[link->to->index][n]);
#endif
                    }
                }
            }
        }
    }

    FILE *f = fopen(out, "w+");
    if (f != NULL)
    {
        agwrite(g, f);
        fclose(f);
    }
    agclose(g);
    for (size_t l = 0; l < net->size; ++l)
        free(nodes[l]);
}
