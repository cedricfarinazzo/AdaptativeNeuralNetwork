#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ANN/tools.h"

double f_init_rand_norm()
{
    return (((double)rand())/RAND_MAX*2.0-1.0);
}

double f_init_input()
{
    return 1.0;
}


double f_act_sigmoid(double n)
{
    return 1/(1 + exp(-n));
}

double f_act_sigmoid_de(double n)
{
    return f_act_sigmoid(n) * (1 - f_act_sigmoid(n));;
}

double f_act_input(double n)
{
    return n;
}

double f_act_input_de(double n __attribute__((unused)))
{
    return 1;
}

double f_act_relu(double n)
{
    return n >= 0 ? n : 0;
}

double f_act_relu_de(double n)
{
    return n >= 0 ? 1 : 0;
}

double f_act_softplus(double n)
{
    return log10(1 + exp(n));
}

double f_act_softplus_de(double n)
{
    return f_act_sigmoid_de(n);
}

#define F_ACT_ELU_ALPHA 0.01

double f_act_elu(double n)
{
    return n >= 0 ? n : F_ACT_ELU_ALPHA * (exp(n) - 1);
}

double f_act_elu_de(double n)
{
    return n >= 0 ? 1 : f_act_elu(n) + F_ACT_ELU_ALPHA;
}

double f_cost_quadratic_loss_de(double o, double t)
{
    return o - t;
}
