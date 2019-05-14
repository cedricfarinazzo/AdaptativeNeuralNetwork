/**
 * \file ANN/models/PCFNN/network.h
 * \brief PCFNN_NETWORK
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 *
 * network unit for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_NETWORK_H_
#define _ANN_MODELS_PCFNN_NETWORK_H_

#include <stdlib.h>
#include <stdio.h>
#include "layer.h"


/**
 * \struct PCFNN_NETWORK
 * \brief Network unit
 *
 * PCFNN_NETWORK: Network unit: It contains an array of PCFNN_LAYER pointer and some metadata.
 */
struct PCFNN_NETWORK {
    size_t size;
    struct PCFNN_LAYER **layers;
    struct PCFNN_LAYER *inputl, *outputl;
};


/**
 * \fn PCFNN_NETWORK_new()
 * \brief Initialize a PCFNN_NETWORK
 * \return PCFNN_NETWORK structure pointer or NULL if an error occured
 */
struct PCFNN_NETWORK *PCFNN_NETWORK_new();


/**
 * \fn PCFNN_NETWORK_free(struct PCFNN_NETWORK *net)
 * \brief Free all memory allocation of an PCFNN_NETWORK (It will call PCFNN_LAYER_free)
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK to free
 */
void PCFNN_NETWORK_free(struct PCFNN_NETWORK *net);


/**
 * \fn PCFNN_NETWORK_clear(struct PCFNN_NETWORK *net)
 * \brief Clear all layers in the PCFNN_NETWORK net (It will call PCFNN_LAYER_clear)
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK to clear
 */
void PCFNN_NETWORK_clear(struct PCFNN_NETWORK *net);


/**
 * \fn PCFNN_NETWORK_addl(struct PCFNN_NETWORK *net, struct PCFNN_LAYER *l)
 * \brief Add the PCFNN_LAYER l to the PCFNN_NETWORK net
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER to add
 * \return 0 if done, 1 if wrong arguments or -1 if an allocation failed and the network is broken
 */
int PCFNN_NETWORK_addl(struct PCFNN_NETWORK *net, struct PCFNN_LAYER *l);


/**
 * \fn PCFNN_NETWORK_build(struct PCFNN_NETWORK *net)
 * \brief Initialize all internal data of the PCFNN_NETWORK l and build of PCFNN_LAYER it contains
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \return 0 if done, 1 if wrong arguments or -1 if an allocation failed and the network and all layers it contains are broken
 */
int PCFNN_NETWORK_build(struct PCFNN_NETWORK *net);


/**
 * \fn PCFNN_NETWORK_get_ram_usage(struct PCFNN_NETWORK *net)
 * \brief Give the number of bytes used by the PCFNN_NETWORK net and all PCFNN_LAYER it contains
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \return (size_t) number of bytes if l is NULL return 0
 */
size_t PCFNN_NETWORK_get_ram_usage(struct PCFNN_NETWORK *net);


void PCFNN_NETWORK_summary(struct PCFNN_NETWORK *net, size_t param[5]);


void PCFNN_NETWORK_print_summary(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_NETWORK_H_ */
