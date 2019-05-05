/**
 * \file ANN/models/PCFNN/backprop.h
 * \brief PCFNN_BACKPROP
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 *
 * Backpropagation functions for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_BACKPROP_H
#define _ANN_MODELS_PCFNN_BACKPROP_H

#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


/**
 * \fn PCFNN_NETWORK_backprop
 * \brief Run the Backpropagation Algorithm on the network net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] target (double*) double array: expected output of the output layer of net
 * \param[in] eta (double) learning rate
 * \param[in] alpha (double) momentum rate
 * \param[in] f_cost (double(*f_cost)(double, double)) a cost function pointer
 */
void PCFNN_NETWORK_backprop(struct PCFNN_NETWORK *net, double *target, double eta, double alpha, double(*f_cost)(double, double));


/**
 * \fn PCFNN_NETWORK_apply_delta
 * \brief Apply all delta calculated by PCFNN_NETWORK_backprop on the network net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 */
void PCFNN_NETWORK_apply_delta(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_BACKPROP_H */
