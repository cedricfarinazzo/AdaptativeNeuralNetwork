/**
 * \file ANN/tools.h
 * \brief Some useful functions
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 9 may 2019
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
 * \fn f_init_rand_norm()
 * \brief Weight and bias initialization function for hidden and output layer
 * \return a double between -1 and 1
 */
double f_init_rand_norm();

/**
 * \fn f_init_input()
 * \brief Weight and bias initialization function input layer
 * \return 1
 */
double f_init_input();


// Activation functions

/**
 * \fn f_act_sigmoid(double n)
 * \brief Sigmoid activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_sigmoid(double n);

/**
 * \fn f_act_sigmoid_de(double n)
 * \brief Derivative sigmoid activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_sigmoid_de(double n);

/**
 * \fn f_act_input(double n)
 * \brief Activation function for input layer (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_input(double n);

/**
 * \fn f_act_input_de(double n)
 * \brief Derivative activation function for input layer (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_input_de(double n);

/**
 * \fn f_act_relu(double n)
 * \brief ReLu activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_relu(double n);

/**
 * \fn f_act_relu_de(double n)
 * \brief Derivative ReLu activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_relu_de(double n);

/**
 * \fn f_act_softplus(double n)
 * \brief SoftPlus activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_softplus(double n);

/**
 * \fn f_act_softplus_de(double n)
 * \brief Derivative SoftPlus activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_softplus_de(double n);

/**
  \def F_ACT_ELU_ALPHA
  Elu function constant: default 0.01
*/
#ifndef F_ACT_ELU_ALPHA
#define F_ACT_ELU_ALPHA 0.01
#endif /* F_ACT_ELU_ALPHA */

/**
 * \fn f_act_elu(double n)
 * \brief Elu activation function (for feedforward algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_elu(double n);

/**
 * \fn f_act_elu_de(double n)
 * \brief Derivative Elu activation function (for backpropagation algorithm)
 * \param[in] n activation sum
 * \return double
 */
double f_act_elu_de(double n);


// Cost functions

/**
  \def F_COST_QUADRATIC_CONSTANT
  Quadratic loss function constant: default 1/2
*/
#ifndef F_COST_QUADRATIC_CONSTANT
#define F_COST_QUADRATIC_CONSTANT 1/2
#endif /* F_COST_QUADRATIC_CONSTANT */

/**
 * \fn f_cost_quadratic_loss(double o, double t)
 * \brief Quadratic cost function
 * \param[in] o output
 * \param[in] t target
 * \return double
 */
double f_cost_quadratic_loss(double o, double t);

/**
 * \fn f_cost_quadratic_loss_de(double o, double t)
 * \brief Derivative Quadratic cost function(for backpropagation algorithm)
 * \param[in] o output
 * \param[in] t target
 * \return double
 */
double f_cost_quadratic_loss_de(double o, double t);

#endif /* _ANN_TOOLS_H_ */
