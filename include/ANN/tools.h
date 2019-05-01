#ifndef _ANN_TOOLS_H_
#define _ANN_TOOLS_H_

#include <stdlib.h>
#include <math.h>
#include <time.h>

double f_init_rand_norm();

double f_init_input();


double f_act_sigmoid(double n);

double f_act_sigmoid_de(double n);

double f_act_input(double n);

double f_act_input_de(double n);

double f_act_relu(double n);

double f_act_relu_de(double n);

double f_act_softplus(double n);

double f_act_softplus_de(double n);

double f_act_elu(double n);

double f_act_elu_de(double n);

double f_cost_quadratic_loss_de(double o, double t);

#endif /* _ANN_TOOLS_H_ */
