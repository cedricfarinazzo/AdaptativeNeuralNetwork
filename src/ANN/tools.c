#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "tools.h"

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
    return 1/(1+exp(-n));
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
