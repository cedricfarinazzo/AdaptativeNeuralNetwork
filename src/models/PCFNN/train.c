#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ANN/models/PCFNN/neuron.h"
#include "ANN/models/PCFNN/layer.h"
#include "ANN/models/PCFNN/network.h"
#include "ANN/models/PCFNN/feedforward.h"
#include "ANN/models/PCFNN/backprop.h"
#include "ANN/models/PCFNN/batch.h"

#include "ANN/models/PCFNN/train.h"


void __PCFNN_BATCH_shuffle(size_t *array, size_t n)
{
    for (size_t i = 0; i < n - 1; i++)
    {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        size_t t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}


double *PCFNN_NETWORK_train(struct PCFNN_NETWORK *net, double **data, double **target,
                            size_t size, double validation_split,
                            int shuffle, unsigned long batch_size, size_t epochs, double eta, double alpha,
                            double(*f_cost)(double, double), double(*f_cost_de)(double, double)
                            , double *status)
{
    if (net == NULL || data == NULL || target == NULL || size == 0 || validation_split < 0 || validation_split > 1)
        return NULL;
    if (validation_split == 0) {
        if (batch_size == 0 || epochs == 0 || eta == 0 || f_cost_de == NULL)
            return NULL;
    } else {
        if (f_cost == NULL)
            return NULL;
    }

    size_t validationsize = (size_t)floor(size * validation_split);
    size_t trainingsize = size - validationsize;

    double __status; if (status == NULL) status = &__status;
    *status = 0;

    size_t order[size];
    for(size_t i = 0; i < size; ++i) order[i] = i;
    __PCFNN_BATCH_shuffle(order, size);
    if (validation_split != 1) {
        double stepstatus = (1/(double)(epochs * trainingsize)) * 100;
        PCFNN_NETWORK_init_batch(net);

        for(size_t e = 0; e < epochs; ++e)
        {
            if (shuffle)
                __PCFNN_BATCH_shuffle(order, trainingsize);
            for(size_t i = 0; i < trainingsize; i+=batch_size)
            {
                for(size_t j = 0; j < batch_size && i+j < trainingsize; ++j, (*status) += stepstatus)
                {
                    PCFNN_NETWORK_feedforward(net, data[order[i+j]]);
                    PCFNN_NETWORK_backprop(net, target[order[i+j]], eta, alpha, f_cost_de);
                }
                PCFNN_NETWORK_apply_delta(net);
                PCFNN_NETWORK_clear_batch(net);
            }
        }

        PCFNN_NETWORK_free_batch(net);
    }
    *status = 100.0;

    if (validation_split == 0)
        return NULL;
    double *err = calloc(net->outputl->size, sizeof(double));
    for(size_t i = trainingsize; i < size; ++i)
    {
        PCFNN_NETWORK_feedforward(net, data[order[i]]);
        for(size_t j = 0; j < net->outputl->size; ++j)
            err[j] += f_cost(net->outputl->neurons[j]->output, target[order[i]][j]);
    }
    for(size_t i = 0; i < net->outputl->size; ++i)
        err[i] /= (double)(size - trainingsize);
    return err;
}
