/*
 * test_core_neuron.c
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <criterion/criterion.h>

#include "../ANN/core/neuron.h"

void setup() {
    srand(time(NULL));
}

double f_init()
{
    return (((double)rand())/RAND_MAX*2.0-1.0)/2.0;
}

double f_act(double n)
{
    return 1/(1+exp(-n));
}

Test(NEURON, Init)
{
    size_t s = 2;
    struct NEURON *n = NEURON_new(s, f_init, f_act);
    cr_expect_not_null(n);
    cr_expect_eq(n->size, s);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);
    
    NEURON_free(n);
}

Test(NEURON, WrongInit)
{
    size_t s = 0;
    struct NEURON *n = NEURON_new(s, f_init, NULL);
    cr_expect_null(n);
    NEURON_free(n);
}

Test(NEURON, feedforward)
{
    size_t s = 2;
    struct NEURON *n = NEURON_new(s, f_init, NULL);
    
    double inputs[] = {2.5, 3.14159};
    cr_expect_neq(NEURON_feedforward(n, inputs, f_act), 0);
    NEURON_clear(n);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);

    NEURON_free(n);
}

Test(NEURON, feedforward2)
{
    size_t s = 2;
    struct NEURON *n = NEURON_new(s, f_init, f_act);
    
    double inputs[] = {2.5, 3.14159};
    cr_expect_neq(NEURON_feedforward(n, inputs, NULL), 0);
    NEURON_clear(n);
    cr_expect_eq(n->output, 0);
    cr_expect_eq(n->activation, 0);
    cr_expect_eq(n->delta, 0);

    NEURON_free(n);
}

Test(NEURON, Clone_stat)
{
    size_t s = 3;
    struct NEURON *n = NEURON_new(s, f_init, NULL);
    
    struct NEURON *b = NEURON_clone_stat(n);
    cr_expect_not_null(b);
    cr_expect_eq(b->size, n->size);
    cr_expect_eq(b->f_init, n->f_init);
    cr_expect_eq(b->f_act, n->f_act);
    
    cr_expect_neq(b->bias, n->bias);
    
    NEURON_free(n);
    NEURON_free(b);
}

Test(NEURON, Clone_all)
{
    size_t s = 42;
    struct NEURON *n = NEURON_new(s, f_init, f_act);
    
    struct NEURON *b = NEURON_clone_all(n);
    cr_expect_not_null(b);
    cr_expect_eq(b->size, n->size);
    cr_expect_eq(b->f_init, n->f_init);
    cr_expect_eq(b->f_act, n->f_act);
    
    cr_expect_eq(b->bias, n->bias);
    cr_expect_eq(b->output, 0);
    cr_expect_eq(b->activation, 0);
    cr_expect_eq(b->delta, 0);
    
    for(size_t i = 0; i < n->size; ++i)
        cr_expect_eq(b->weights[i], n->weights[i]);
    
    NEURON_free(n);
    NEURON_free(b);
}

Test(NEURON, AllocFail)
{
    size_t s = 3141592653589793238;
    struct NEURON *n = NEURON_new(s, f_init, f_act);
    cr_expect_null(n);
    NEURON_free(n);
}

