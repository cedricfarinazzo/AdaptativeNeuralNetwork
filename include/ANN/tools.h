/**
 * \file ANN/tools.h
 * \brief Some useful functions
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 5 may 2019
 *
 * Some useful functions such as activation functions, cost functions or weight/bias initialization functions
 */

#ifndef _ANN_TOOLS_H_
#define _ANN_TOOLS_H_

#include <stdlib.h>
#include <math.h>
#include <time.h>

// Weight/bias initialization functions

/**
 * \fn f_init_rand_norm
 * \brief Weight and bias initialization function for hidden and output layer
 * \return a double between -1 and 1
 */
double f_init_rand_norm();

/**
 * \fn f_init_input
 * \brief Weight and bias initialization function input layer
 * \return 1
 */
double f_init_input();


// Activation functions
/**
 * \fn f_act_sigmoid
 * \brief Sigmoid activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_sigmoid(double n);

/**
 * \fn f_act_sigmoid_de
 * \brief Derivative sigmoid activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_sigmoid_de(double n);

/**
 * \fn f_act_input
 * \brief Activation function for input layer (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_input(double n);

/**
 * \fn f_act_input_de
 * \brief Derivative activation function for input layer (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_input_de(double n);

/**
 * \fn f_act_relu
 * \brief ReLu activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_relu(double n);

/**
 * \fn f_act_relu_de
 * \brief Derivative ReLu activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_relu_de(double n);

/**
 * \fn f_act_softplus
 * \brief SoftPlus activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_softplus(double n);

/**
 * \fn f_act_softplus_de
 * \brief Derivative SoftPlus activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_softplus_de(double n);

/**
 * \fn f_act_elu
 * \brief Elu activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_elu(double n);

/**
 * \fn f_act_elu_de
 * \brief Derivative Elu activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_elu_de(double n);


// Cost functions
/**
 * \fn f_cost_quadratic_loss_de
 * \brief Quadratic cost function (for backpropagation algorithm)
 * \param[in] o output
 * \param[in] t target
 * \return double
 */
double f_cost_quadratic_loss_de(double o, double t);

#endif /* _ANN_TOOLS_H_ */
