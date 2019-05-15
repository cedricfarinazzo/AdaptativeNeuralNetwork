#include <stdlib.h>

#include "ANN/models/CNN/filter.h"

struct CNN_FILTER* CNN_FILTER_new(size_t size, size_t strides, size_t padding,
                                  double(*f_act)(double), double(*f_act_de)(double))
{
    struct CNN_FILTER* filter = malloc(sizeof(struct CNN_FILTER));
    if (filter == NULL) return NULL;
    filter->size = size;
    filter->filter = calloc(size * size, sizeof(double));
    if (filter->filter == NULL) { free(filter); return NULL; }
    filter->strides = strides;
    filter->padding = padding;
    filter->f_act = f_act;
    filter->f_act_de = f_act_de;
    return filter;
}

void CNN_FILTER_free(struct CNN_FILTER *filter)
{
    if (filter != NULL)
    {
        free(filter->filter);
        free(filter);
    }
}

void CNN_FILTER_fill(struct CNN_FILTER *filter, double *array)
{
    if (filter == NULL || array == NULL) return;
    for(size_t i = 0; i < filter->size * filter->size; ++i)
        filter->filter[i] = array[i];
}
