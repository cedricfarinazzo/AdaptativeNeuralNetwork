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


Agraph_t *PCFNN_GRAPH_create_graph(struct PCFNN_NETWORK *net, char *graph_name)
{
    if (net == NULL || graph_name == NULL) return NULL;

    Agraph_t *g;
    Agraph_t *layers[net->size];
    Agnode_t **nodes[net->size];
    
#ifdef WITH_CGRAPH
    g = agopen(graph_name, Agdirected, NULL);
#else
    aginit();
    g = agopen(graph_name, AGDIGRAPH);
#endif

    if (g == NULL) return NULL;
    
    agsafeset(g, "charset", "UTF-8", "");
    agsafeset(g, "rankdir", "LR", "");
    agsafeset(g, "center", "true", "");
    agsafeset(g, "ranksep", "5", "");
    
    for (size_t l = 0; l < net->size; ++l)
    {
        char lbuff[42];
        sprintf(lbuff, "L%ld", net->layers[l]->index);
        layers[l] = agsubg(g, lbuff, 1);
        agsafeset(layers[l], "style", "filled", "");
        agsafeset(layers[l], "bgcolor", "black", "");
        agsafeset(layers[l], "label", lbuff, "");
        
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
            agsafeset(nodes[l][n], "shape", "circle", "");
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

    for (size_t l = 0; l < net->size; ++l)
        free(nodes[l]);
    return g;
}


int PCFNN_GRAPH_create_graph_to_dot_file(struct PCFNN_NETWORK *net, char *graph_name, char *fout)
{
    if (net == NULL || fout == NULL || graph_name == NULL) return -1;

    Agraph_t *g = PCFNN_GRAPH_create_graph(net, graph_name);
    if (g == NULL) return -1;

    FILE *f = fopen(fout, "w+");
    if (f != NULL) {
        agwrite(g, f);
        fclose(f);
        agclose(g);
        return 0;
    } else {
        agclose(g);
        return -1;
    }
}


int PCFNN_GRAPH_render_graph_to_stream(struct PCFNN_NETWORK *net, char *graph_name, char *format, FILE *fout)
{
    if (net == NULL || graph_name == NULL || format == NULL || fout == NULL) return -1;
    
    Agraph_t *g = PCFNN_GRAPH_create_graph(net, graph_name);
    if (g == NULL) return -1;

    GVC_t *gvc;
    gvc = gvContext();
    if (gvc == NULL) { agclose(g); return -1;}
    gvLayout(gvc, g, "dot");

    gvRender (gvc, g, format, fout);

    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);

    return 0;
}


int PCFNN_GRAPH_render_graph_to_file(struct PCFNN_NETWORK *net, char *graph_name, char *format, char *fout)
{
    if (net == NULL || graph_name == NULL || format == NULL || fout == NULL) return -1;
    
    Agraph_t *g = PCFNN_GRAPH_create_graph(net, graph_name);
    if (g == NULL) return -1;

    GVC_t *gvc;
    gvc = gvContext();
    if (gvc == NULL) { agclose(g); return -1; }
    gvLayout(gvc, g, "dot");

    gvRenderFilename (gvc, g, format, fout);

    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);

    return 0;
}
