/**
 * \file ANN/models/CNN/filter.h
 * \brief CNN_FILTER
 * \author Cedric FARINAZZO
 * \version 0.1
 * \date 15 may 2019
 *
 * CNN neural network
 */

#ifndef _ANN_MODELS_CNN_FILTER_H_
#define _ANN_MODELS_CNN_FILTER_H_

#include <stdlib.h>


struct CNN_FILTER {
    size_t size;
    double *filter;
    size_t strides;
    size_t padding;
    double(*f_act)(double);
    double(*f_act_de)(double);
};

#endif /* _ANN_MODELS_CNN_FILTER_H_ */
