/**
 * \file ANN/models/PCFNN/feedforward.h
 * \brief PCFNN_FEEDFORWARD
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 *
 * Feedforward functions for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_FEEDFORWARD_H
#define _ANN_MODELS_PCFNN_FEEDFORWARD_H

#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


/**
 * \fn PCFNN_NEURON_feedforward
 * \brief Feedforward the neuron n
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \param[in] inputs (double*) a double array to be used to populate the neuron
 * \param[in] f_act (double(*f_act)(double)) a pointer on activation functon or NULL
 * \param[in] f_act_de (double(*f_act_de)(double)) a pointer on derivative activation functon who is the derivate of the f_act function pointer or NULL
 * \return neuron output
 */
double PCFNN_NEURON_feedforward(struct PCFNN_NEURON *n, double *inputs, double(*f_act)(double), double(*f_act_de)(double));


/**
 * \fn PCFNN_LAYER_feedforward_input
 * \brief Feedforward the input layer l
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \param[in] inputs (double*) a double array to be used to populate the layer
 */
void PCFNN_LAYER_feedforward_input(struct PCFNN_LAYER *l, double *inputs);


/**
 * \fn PCFNN_LAYER_feedforward
 * \brief Feedforward the hidden layer l
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 */
void PCFNN_LAYER_feedforward(struct PCFNN_LAYER *l);


/**
 * \fn PCFNN_NETWORK_feedforward
 * \brief Feedforward the PCFNN_NETWORK net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] inputs (double*) a double array to be used to populate the input layer
 */
void PCFNN_NETWORK_feedforward(struct PCFNN_NETWORK *net, double *inputs);


/**
 * \fn PCFNN_NETWORK_get_output
 * \brief Return a double array which is the output of the output layer of net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \return a double array or NULL if an error occured
 */
double *PCFNN_NETWORK_get_output(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_FEEDFORWARD_H */
