/**
 * \file ANN/models/PCFNN/neuron.h
 * \brief PCFNN_NEURON
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 * 
 * Neuron unit for PCFNN neural network
 */

#ifndef _ANN_MODELS_PCFNN_NEURON_H_
#define _ANN_MODELS_PCFNN_NEURON_H_

#include "../../config.h"
#include <stdlib.h>


/**
 * \enum PCFNN_NEURON_LOCK_STATE
 * \brief Lock state of a neuron
 */
enum PCFNN_NEURON_LOCK_STATE {
    PCFNN_NEURON_LOCK,
    PCFNN_NEURON_UNLOCK,
};


/**
 * \struct PCFNN_NEURON
 * \brief Neuron unit
 *
 * PCFNN_NEURON: neuron unit. It contains the input size of the neuron, the bias, the weights array
 * and some internal data such as activation functon or the activation sum. 
 */
typedef struct PCFNN_NEURON {
    size_t size;
    double *weights;
    double bias, output; 
    // INTERNAL
    double activation, delta, bdelta, dsum;
    double *wdelta, *lastdw;
    struct PCFNN_NEURON **inputs; 
    double(*f_init)();
    double(*f_act)(double);
    double(*f_act_de)(double);
    enum PCFNN_NEURON_LOCK_STATE state;
} PCFNN_NEURON;


/**
 * \fn PCFNN_NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double))
 * \brief Initialize a PCFNN_NEURON
 * \param[in] size (size_t) input size
 * \param[in] f_init (double(*f_init)()) a pointer on an weights/bias initialisation functon or NULL
 * \param[in] f_act (double(*f_act)(double)) a pointer on activation functon or NULL
 * \param[in] f_act_de (double(*f_act_de)(double)) a pointer on derivative activation functon who is the derivate of the f_act function pointer or NULL
 * \return PCFNN_NEURON structure pointer or NULL if an error occured
 */
struct PCFNN_NEURON *PCFNN_NEURON_new(size_t size, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double));


/**
 * \fn PCFNN_NEURON_clear(struct PCFNN_NEURON *n)
 * \brief Clear a PCFNN_NEURON
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 */
void PCFNN_NEURON_clear(struct PCFNN_NEURON *n);


/**
 * \fn PCFNN_NEURON_free(struct PCFNN_NEURON *n)
 * \brief Free all memory allocation of an PCFNN_NEURON
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON to free
 */
void PCFNN_NEURON_free(struct PCFNN_NEURON *n);


/**
 * \fn PCFNN_NEURON_addinputs(struct PCFNN_NEURON *n, size_t inputs)
 * \brief Increase the input size of the PCFNN_NEURON
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 * \param[in] inputs (size_t) number of input to add. if 0 do nothing.
 */
void PCFNN_NEURON_addinputs(struct PCFNN_NEURON *n, size_t inputs);


/**
 * \fn PCFNN_NEURON_build(struct PCFNN_NEURON *n)
 * \brief Initialize all internal data of a PCFNN_NEURON
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 */
void PCFNN_NEURON_build(struct PCFNN_NEURON *n);


/**
 * \fn PCFNN_NEURON_get_ram_usage(struct PCFNN_NEURON *n)
 * \brief Give the number of bytes used by the PCFNN_NEURON n
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 * \return (size_t) number of bytes
 */
size_t PCFNN_NEURON_get_ram_usage(struct PCFNN_NEURON *n);


/**
 * \fn PCFNN_NEURON_clone(struct PCFNN_NEURON *n)
 * \brief Copy the input size, initialisation function and activation function from n and create a new PCFNN_NEURON
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 * \return (struct PCFNN_NEURON*) the new neuron or NULL if an error occured
 */
struct PCFNN_NEURON *PCFNN_NEURON_clone(struct PCFNN_NEURON *n);


/**
 * \fn PCFNN_NEURON_clone_all(struct PCFNN_NEURON *n)
 * \brief Copy all data from n and create a new PCFNN_NEURON
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 * \return (struct PCFNN_NEURON*) the new neuron or NULL if an error occured
 */
struct PCFNN_NEURON *PCFNN_NEURON_clone_all(struct PCFNN_NEURON *n);


/**
 * \fn PCFNN_NEURON_set_state_lock(struct PCFNN_NEURON *n, enum PCFNN_NEURON_LOCK_STATE state)
 * \brief Set lock state of the neuron n
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 * \param[in] state (enum PCFNN_NEURON_LOCK_STATE) lock state
 */
void PCFNN_NEURON_set_state_lock(struct PCFNN_NEURON *n, enum PCFNN_NEURON_LOCK_STATE state);


/**
 * \fn PCFNN_NEURON_summary(struct PCFNN_NEURON *n, size_t param[2])
 * \brief Write on param the number of unlocked parameters and locked parameters
 * \param[in] n (struct PCFNN_NEURON*) a pointer an a PCFNN_NEURON
 * \param[out] param (size_t) param[0] will be the number of unlocked parameters and param[1] the number of locked parameters
 */
void PCFNN_NEURON_summary(struct PCFNN_NEURON *n, size_t param[2]);


#endif /* _ANN_MODELS_PCFNN_NEURON_H_ */
