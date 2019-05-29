/**
 * \file ANN/models/PCFNN/graph.h
 * \brief PCFNN_GRAPH
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 24 may 2019
 *
 * Graph functions for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_GRAPH_H
#define _ANN_MODELS_PCFNN_GRAPH_H

#include <stdlib.h>
#include <stdio.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"

#include <graphviz/gvc.h>
#ifdef WITH_CGRAPH
#include <graphviz/cgraph.h>
#else
#include <graphviz/graph.h>
#endif

/**
 * \fn Agraph_t *PCFNN_GRAPH_create_graph(struct PCFNN_NETWORK *net, char *graph_name)
 * \brief Generate a graph (with graphviz) from the given PCFNN_NETWORK
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] graph_name (char*) name of your network
 * \return Agraph_t*: graphviz graph pointer
 */
Agraph_t *PCFNN_GRAPH_create_graph(struct PCFNN_NETWORK *net, char *graph_name);


/**
 * \fn int PCFNN_GRAPH_create_graph_to_dot_file(struct PCFNN_NETWORK *net, char *graph_name, char *fout)
 * \brief Generate a graph (with graphviz) from the given PCFNN_NETWORK and write it to fout with the dot file format
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] graph_name (char*) name of your network
 * \param[in] fout (char*) path to file
 * \return 0 if ok otherwise -1
 */
int PCFNN_GRAPH_create_graph_to_dot_file(struct PCFNN_NETWORK *net, char *graph_name, char *fout);


/**
 * \fn int PCFNN_GRAPH_render_graph_to_stream(struct PCFNN_NETWORK *net, char *graph_name, char *format, FILE *fout)
 * \brief Generate a graph (with graphviz) from the given PCFNN_NETWORK and render it to the given stream fout with the given format
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] graph_name (char*) name of your network
 * \param[in] format (char*) name of the renderer to use (ex: png, ps, dot, ...)
 * \param[in] fout (FILE*) stream to file with written permission
 * \return 0 if ok otherwise -1
 */
int PCFNN_GRAPH_render_graph_to_stream(struct PCFNN_NETWORK *net, char *graph_name, char *format, FILE *fout) __attribute__((deprecated));


/**
 * \fn int PCFNN_GRAPH_render_graph_to_file(struct PCFNN_NETWORK *net, char *graph_name, char *format, char *fout)
 * \brief Generate a graph (with graphviz) from the given PCFNN_NETWORK and render it to the given path fout with the given format
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] graph_name (char*) name of your network
 * \param[in] format (char*) name of the renderer to use (ex: png, ps, dot, ...)
 * \param[in] fout (char*) path to file (will be created if not already exist)
 * \return 0 if ok otherwise -1
 */
int PCFNN_GRAPH_render_graph_to_file(struct PCFNN_NETWORK *net, char *graph_name, char *format, char *fout) __attribute__((deprecated));

#endif /* _ANN_MODELS_PCFNN_GRAPH_H */
