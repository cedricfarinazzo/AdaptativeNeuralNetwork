#ifndef _ANN_MODELS_PCFNN_FILE_H
#define _ANN_MODELS_PCFNN_FILE_H

#include <stdlib.h>
#include <stdio.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


int PCFNN_NETWORK_save_conf(struct PCFNN_NETWORK *net, FILE *fout);


int PCFNN_NETWORK_load_conf(struct PCFNN_NETWORK *net, FILE *fin);

#endif /* _ANN_MODELS_PCFNN_FILE_H */
