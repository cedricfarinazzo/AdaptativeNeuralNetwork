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

#include "../../config.h"
#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "ANN/tools.h"


/**
 * \struct PCFNN_NETWORK
 * \brief Network unit
 *
 * PCFNN_NETWORK: Network unit: It contains an array of PCFNN_LAYER pointer and some metadata.
 */
typedef struct PCFNN_NETWORK {
    size_t size;
    struct PCFNN_LAYER **layers;
    struct PCFNN_LAYER *inputl, *outputl;
} PCFNN_NETWORK;


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
 * \fn PCFNN_NETWORK *PCFNN_NETWORK_build_from_array(size_t *spec, size_t len, double(*f_init)(double), double(*f_act)(double), double(*f_act_de)(double))
 * \brief Initialize a new PCFNN_NETWORK from an array of number that represent the number of neurons for each layers
 * \param[in] spec (size_t*) array of number that represent the number of neurons for each layers
 * \param[in] len (size_t) length of spec
 * \param[in] f_init (double(*f_init)()) a pointer on an weights/bias initialisation functon
 * \param[in] f_act (double(*f_act)(double)) a pointer on activation functon
 * \param[in] f_act_de (double(*f_act_de)(double)) a pointer on derivative activation functon who is the derivate of the f_act function pointer
 * \return a new PCFNN_NETWORK or NULL if an error occured
 */
struct PCFNN_NETWORK *PCFNN_NETWORK_build_from_array(size_t *spec, size_t len, double(*f_init)(double), double(*f_act)(double), double(*f_act_de)(double));


/**
 * \fn PCFNN_NETWORK_get_ram_usage(struct PCFNN_NETWORK *net)
 * \brief Give the number of bytes used by the PCFNN_NETWORK net and all PCFNN_LAYER it contains
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \return (size_t) number of bytes if l is NULL return 0
 */
size_t PCFNN_NETWORK_get_ram_usage(struct PCFNN_NETWORK *net);


/**
 * \fn PCFNN_NETWORK_summary(struct PCFNN_NETWORK *net, size_t param[5])
 * \brief Write on param network statistics
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 * \param[out] param (size_t) network statistics\n
 *  param[0]: number of unlocked parameters\n
 *  param[1]: number of locked parameters\n
 *  param[2]: ram usage in bytes\n
 *  param[3]: number of layer\n
 *  param[4]: number of neurons
 */

void PCFNN_NETWORK_summary(struct PCFNN_NETWORK *net, size_t param[5]);


/**
 * \fn PCFNN_NETWORK_print_summary(struct PCFNN_NETWORK *net)
 * \brief Print neural network summary
 * \param[in] net (struct PCFNN_NETWORK*) a pointer an a PCFNN_NETWORK
 */

void PCFNN_NETWORK_print_summary(struct PCFNN_NETWORK *net);

#endif /* _ANN_MODELS_PCFNN_NETWORK_H_ */
