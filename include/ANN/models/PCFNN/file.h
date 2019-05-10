/**
 * \file ANN/models/PCFNN/file.h
 * \brief PCFNN_FILE
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 *
 * IO functions for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_FILE_H
#define _ANN_MODELS_PCFNN_FILE_H

#include <stdlib.h>
#include <stdio.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


/**
 * \fn PCFNN_NETWORK_save_conf(struct PCFNN_NETWORK *net, FILE *fout)
 * \brief Save bias and weights of each neurons of net in fout
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] fout (FILE*) stream on file with write permission
 * \return 1 if done or -1 if an error occured
 */
int PCFNN_NETWORK_save_conf(struct PCFNN_NETWORK *net, FILE *fout);


/**
 * \fn PCFNN_NETWORK_load_conf(struct PCFNN_NETWORK *net, FILE *fin)
 * \brief Load bias and weights of each neurons of net from fin
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] fin (FILE*) stream on file with read permission
 * \return 1 if done or -1 if an error occured
 */
int PCFNN_NETWORK_load_conf(struct PCFNN_NETWORK *net, FILE *fin);

#endif /* _ANN_MODELS_PCFNN_FILE_H */
