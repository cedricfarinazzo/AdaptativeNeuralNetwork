// File : pyann_low.i
%module pyann_low


// File : version
%inline %{
#include "ANN/version.h"

int ann_version_major = ANN_VERSION_MAJOR;
int ann_version_minor = ANN_VERSION_MINOR;
int ann_version_patch = ANN_VERSION_PATCH;
int ann_version_tweak = ANN_VERSION_TWEAK;
char *ann_version = ANN_VERSION;
%}


// File : tools
%inline %{
#include "ANN/tools.h"

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
double f_act_swish(double n);
double f_act_swish_de(double n);

double f_cost_quadratic_loss(double o, double t);
double f_cost_quadratic_loss_de(double o, double t);
%}

// File : ANN.models.models

// File : ANN.models.PCFNN.PCFNN

// File : ANN.models.PCFNN.neuron
%inline %{
#include "ANN/models/PCFNN/neuron.h"
%}

%extend PCFNN_NEURON {
    PCFNN_NEURON __init__(size_t size, double(*f_init)(), double(*f_act)(double), double(*f_act_de)(double)) {
        return (PCFNN_NEURON*)PCFNN_NEURON_new(size, f_init, f_act, f_act_de);
    }

    void __del__() {
        PCFNN_NEURON_free($self);
    }

    void clear() {
        PCFNN_NEURON_clear($self);
    }
}


