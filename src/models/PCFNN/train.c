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


int PCFNN_NETWORK_train(struct PCFNN_NETWORK *net, double **data, double **target,
                         size_t size, double validation_split, int(*f_val)(double, double),
                         int shuffle, unsigned long batch_size, size_t epochs, double eta, double alpha, 
                         double(*f_cost)(double, double), double *status)
{
    if (net == NULL || data == NULL || target == NULL || batch_size <= 0 || epochs == 0
        || validation_split < 0 || validation_split > 1 || f_cost == NULL 
        || (validation_split < 0 && f_val == NULL) || size < batch_size)
        return -1;
    size_t validationsize = (size_t)floor(size * validation_split);
    size_t trainingsize = size - validationsize;
    
    double __status; if (status == NULL) status = &__status;
    double stepstatus = (1/(double)(epochs * trainingsize)) * 100;
    *status = 0;

    size_t trainingorder[trainingsize];
    for(size_t i = 0; i < trainingsize; ++i)
        trainingorder[i] = i;
    
    PCFNN_NETWORK_init_batch(net);

    for(size_t e = 0; e < epochs; ++e)
    {
        if (shuffle)
            __PCFNN_BATCH_shuffle(trainingorder, trainingsize);
        for(size_t i = 0; i < trainingsize; i+=batch_size)
        {
            for(size_t j = 0; j < batch_size && i+j < trainingsize; ++j, (*status) += stepstatus)
            {
                PCFNN_NETWORK_feedforward(net, data[trainingorder[i+j]]);
                PCFNN_NETWORK_backprop(net, target[trainingorder[i+j]], eta, alpha, f_cost);
            }
            PCFNN_NETWORK_apply_delta(net);
            PCFNN_NETWORK_clear_batch(net);
        }
    }
    
    PCFNN_NETWORK_free_batch(net);
    *status = 100.0;

    int ok = 1; 
    for (size_t i = trainingsize; ok && i < size; ++i)
    {
        PCFNN_NETWORK_feedforward(net, data[i]);
        double *out = PCFNN_NETWORK_get_output(net);
        for(size_t j = 0; j < net->outputl->size; ++j)
            ok = f_val(out[j], target[i][j]);
        free(out);
    }

    return ok;
}

