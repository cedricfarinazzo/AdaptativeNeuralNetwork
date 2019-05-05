/**
 * \file ANN/models/PCFNN/batch.h
 * \brief PCFNN_BATCH
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 *
 * IO functions for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_BATCH_H
#define _ANN_MODELS_PCFNN_BATCH_H

#include <stdlib.h>
#include "neuron.h"
#include "layer.h"
#include "network.h"


/**
 * \fn PCFNN_NETWORK_init_batch
 * \brief Initialize batch for the network net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 */
void PCFNN_NETWORK_init_batch(struct PCFNN_NETWORK *net);


/**
 * \fn PCFNN_NETWORK_free_batch
 * \brief Free all memory allocation for the batch of the network net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 */
void PCFNN_NETWORK_free_batch(struct PCFNN_NETWORK *net);


/**
 * \fn PCFNN_NETWORK_clear_batch
 * \brief Partially clear batch data of the network net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 */
void PCFNN_NETWORK_clear_batch(struct PCFNN_NETWORK *net);


/**
 * \fn PCFNN_NETWORK_clear_batch
 * \brief Clear all batch data of the network net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 */
void PCFNN_NETWORK_clear_batch_all(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_BATCH_H */
