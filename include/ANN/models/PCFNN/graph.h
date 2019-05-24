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


void PCFNN_GRAPH_create_graph_to_file(struct PCFNN_NETWORK *net, char *out, char *graph_name);

#endif /* _ANN_MODELS_PCFNN_GRAPH_H */
