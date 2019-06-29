/**
 * \file ANN/models/PCFNN/layer.h
 * \brief PCFNN_LAYER
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 *
 * Layer unit for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_LAYER_H_
#define _ANN_MODELS_PCFNN_LAYER_H_

#include "../../config.h"
#include <stdlib.h>
#include "ANN/tools.h"
#include "neuron.h"


/**
 * \enum PCFNN_LAYER_TYPE
 * \brief Type list of a layer
 */
enum PCFNN_LAYER_TYPE {
    PCFNN_LAYER_INPUT,
    PCFNN_LAYER_HIDDEN,
    PCFNN_LAYER_OUTPUT,
};

struct PCFNN_LAYER;

/**
 * \struct PCFNN_LAYER_LINK
 * \brief Structure to represent a link between to PCFNN_LAYER
 */
struct PCFNN_LAYER_LINK {
    int index_from, index_to;
    struct PCFNN_LAYER *from, *to;
    size_t size_from, size_to;
    double(*f_init_to)();
    double(*f_act_to)(double);
    double(*f_act_de_to)(double);
    int isInit;
    size_t offset_from, offset_to;
};

/**
 * \struct PCFNN_LAYER
 * \brief Layer unit
 *
 * PCFNN_LAYER: Layer unit: It contains an array of PCFNN_NEURON pointer and some metadata.
 */
typedef struct PCFNN_LAYER {
    size_t index;
    size_t size;
    struct PCFNN_NEURON **neurons;
    size_t nblinks;
    struct PCFNN_LAYER_LINK **links;
    double(*f_init)();
    double(*f_act)(double);
    double(*f_act_de)(double);
    enum PCFNN_LAYER_TYPE type;
} PCFNN_LAYER;


/**
 * \fn PCFNN_LAYER_new
 * \brief Initialize a PCFNN_LAYER(double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double))
 * \param[in] f_init (double(*f_init)()) a pointer on an weights/bias initialisation functon or NULL
 * \param[in] f_act (double(*f_act)(double)) a pointer on activation functon or NULL
 * \param[in] f_act_de (double(*f_act_de)(double)) a pointer on derivative activation functon who is the derivate of the f_act function pointer or NULL
 * \return PCFNN_LAYER structure pointer or NULL if an error occured
 */
struct PCFNN_LAYER *PCFNN_LAYER_new(double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double));


/**
 * \fn PCFNN_LAYER_free(struct PCFNN_LAYER *l)
 * \brief Free all memory allocation of an PCFNN_LAYER (It will call PCFNN_NEURON_free)
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER to free
 */
void PCFNN_LAYER_free(struct PCFNN_LAYER *l);


/**
 * \fn PCFNN_LAYER_clear(struct PCFNN_LAYER *l)
 * \brief Clear all neurons in the layer l (It will call PCFNN_NEURON_clear)
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER to clear
 */
void PCFNN_LAYER_clear(struct PCFNN_LAYER *l);


/**
 * \fn PCFNN_LAYER_addn(struct PCFNN_LAYER *l, size_t size, size_t inputs, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double))
 * \brief Add neurons in the PCFNN_LAYER l
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \param[in] size (size_t) number of PCFNN_NEURON to Add
 * \param[in] inputs (size_t) number of input for each new PCFNN_NEURON
 * \param[in] f_init (double(*f_init)()) a pointer on an weights/bias initialisation functon or NULL
 * \param[in] f_act (double(*f_act)(double)) a pointer on activation functon or NULL
 * \param[in] f_act_de (double(*f_act_de)(double)) a pointer on derivative activation functon who is the derivate of the f_act function pointer or NULL
 * \return 0 if done, 1 if wrong arguments or -1 if an allocation failed and the layer is broken
 */
int PCFNN_LAYER_addn(struct PCFNN_LAYER *l, size_t size, size_t inputs, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double));


/**
 * \fn PCFNN_LAYER_new_input(size_t size, double(*f_act)(double), double(*f_act_de)(double))
 * \brief Initialize a PCFNN_LAYER as an input layer
 * \param[in] size (size_t) number of neuron in this layer
 * \param[in] f_act (double(*f_act)(double)) a pointer on activation functon or NULL
 * \param[in] f_act_de (double(*f_act_de)(double)) a pointer on derivative activation functon who is the derivate of the f_act function pointer or NULL
 * \return PCFNN_LAYER structure pointer or NULL if an error occured
 */
struct PCFNN_LAYER *PCFNN_LAYER_new_input(size_t size, double(*f_act)(double), double(*f_act_de)(double));


/**
 * \fn PCFNN_LAYER_connect(struct PCFNN_LAYER *from, struct PCFNN_LAYER *to,
                       size_t size_from, size_t size_to,
                       size_t offset_from, size_t offset_to,
                       double(*f_init_to)(), double(*f_act_to)(double), double(*f_act_de_to)(double))
 * \brief Connect two PCFNN_LAYER. It will create a link between from and to
 * \param[in] from (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \param[in] to (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \param[in] size_from (size_t) number of neuron of from to connect
 * \param[in] size_to (size_t) number of neuron of to to connect (It will create size_to neurons in to if necessary)
 * \param[in] offset_from (size_t) offset of offset_from where you want to start the connection for the from layer
 * \param[in] offset_to (size_t) offset of offset_to where you want to start the connection for the to layer
 * \param[in] f_init_to (double(*f_init_to)()) a pointer on an weights/bias initialisation functon or NULL (Will be use if need to create new neurons in the to layer)
 * \param[in] f_act_to (double(*f_act_to)(double)) a pointer on activation functon or NULL (Will be use if need to create new neurons in the to layer)
 * \param[in] f_act_de_to (double(*f_act_de_to)(double)) a pointer on derivative activation functon who is the derivate of the f_act function pointer or NULL (Will be use if need to create new neurons in the to layer)
 * \return 0 if done, 1 if wrong arguments or -1 if an allocation failed and the layer is broken
 */
int PCFNN_LAYER_connect(struct PCFNN_LAYER *from, struct PCFNN_LAYER *to,
                       size_t size_from, size_t size_to,
                       size_t offset_from, size_t offset_to,
                       double(*f_init_to)(), double(*f_act_to)(double), double(*f_act_de_to)(double));


/**
 * \fn PCFNN_LAYER_build(struct PCFNN_LAYER *l)
 * \brief Initialize all internal data of the layer l and build of PCFNN_NEURON it contains
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \return 0 if done, 1 if wrong arguments or -1 if an allocation failed and the layer is broken
 */
int PCFNN_LAYER_build(struct PCFNN_LAYER *l);


/**
 * \fn PCFNN_LAYER_get_ram_usage(struct PCFNN_LAYER *l)
 * \brief Give the number of bytes used by the PCFNN_LAYER l and all PCFNN_NEURON it contains
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \return (size_t) number of bytes if l is NULL return 0
 */
size_t PCFNN_LAYER_get_ram_usage(struct PCFNN_LAYER *l);


/**
 * \fn PCFNN_LAYER_set_lock_state(struct PCFNN_LAYER *l, enum PCFNN_NEURON_LOCK_STATE state, size_t size, size_t offset)
 * \brief Set lock state of size neurons of l starting by offset neuron of l
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \param[in] state (enum PCFNN_NEURON_LOCK_STATE) lock state
 * \param[in] size (size_t) number of neurons to set lock state
 * \param[in] offset (size_t) neurons offset
 */
void PCFNN_LAYER_set_lock_state(struct PCFNN_LAYER *l, enum PCFNN_NEURON_LOCK_STATE state, size_t size, size_t offset);


/**
 * \fn PCFNN_LAYER_summary(struct PCFNN_LAYER *l, size_t param[2])
 * \brief Write on param the number of unlocked parameters and locked parameters
 * \param[in] l (struct PCFNN_LAYER*) a pointer an a PCFNN_LAYER
 * \param[out] param (size_t) param[0] will be the number of unlocked parameters and param[1] the number of locked parameters
 */
void PCFNN_LAYER_summary(struct PCFNN_LAYER *l, size_t param[2]);

#endif /* _ANN_MODELS_PCFNN_LAYER_H_ */
