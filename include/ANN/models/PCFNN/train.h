/**
 * \file ANN/models/PCFNN/train.h
 * \brief PCFNN_TRAIN
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 9 may 2019
 *
 * Train functions for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_TRAIN_H
#define _ANN_MODELS_PCFNN_TRAIN_H

#include <stdlib.h>
#include <math.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"
#include "feedforward.h"
#include "backprop.h"
#include "batch.h"


/**
 * \fn PCFNN_NETWORK_train(struct PCFNN_NETWORK *net, double **data, double **target,
                            size_t size, double validation_split,
                            int shuffle, unsigned long batch_size, size_t epochs, double eta, double alpha,
                            double(*f_cost)(double, double), double(*f_cost_de)(double, double)
                            , double *status)
 * \brief Train the network net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] data (double**) double array array: an array of input data (lenght of data is size)
 * \param[in] target (double**) double array array : an array of expected output of the output layer of net (lenght of target is size)
 * \param[in] size (size_t) lenght of data and target
 * \param[in] validation_split (double) a double between 0 and 1: the part of the dataset to use to test the network
 * \param[in] shuffle (int) 1 to enable shuffle mode and 0 to disable
 * \param[in] batch_size (unsigned long) batch size
 * \param[in] epochs (size_t) number of epochs to run
 * \param[in] eta (double) learning rate
 * \param[in] alpha (double) momentum rate
 * \param[in] f_cost (double(*f_cost)(double, double)) a cost function pointer
 * \param[in] f_cost_de (double(*f_cost_de)(double, double)) a pointer to the f_cost derivative
 * \param[out] status (double*) a pointer on a double. Can be NULL. It will contain the percentage of completion of the training. Usefull with thread
 * \return double array of net->outputl->size elements which is the current error for each output (must be free after usage) if validation_split > 0 else return NULL. NULL if wrong parameters.
 */
double *PCFNN_NETWORK_train(struct PCFNN_NETWORK *net, double **data, double **target,
                            size_t size, double validation_split,
                            int shuffle, unsigned long batch_size, size_t epochs, double eta, double alpha,
                            double(*f_cost)(double, double), double(*f_cost_de)(double, double)
                            , double *status);

#endif /* _ANN_MODELS_PCFNN_TRAIN_H */
