#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "math.h"
#include "node.h"
#include "connection.h"
#include "network.h"

static void shuffle(unsigned int *a, unsigned int n)
{

    unsigned int i;

    for (i = 0; i < n - 1; i++)
    {

        unsigned int j = i + rand() / (RAND_MAX / (n - i) + 1);
        unsigned int t = a[j];

        a[j] = a[i];
        a[i] = t;

    }

}

static void train(struct network *network, double epocs, double learningrate, unsigned int *order, double *inputs, double *outputs, unsigned int sets)
{

    struct nodelayer *inputlayer = network_getnodelayer(network, 0);
    struct nodelayer *outputlayer = network_getnodelayer(network, network->nsize - 1);
    unsigned int epoch;

    for (epoch = 0; epoch < epocs; epoch++)
    {

        unsigned int i;

        shuffle(order, sets);

        for (i = 0; i < sets; i++)
        {

            unsigned int setindex = order[i];
            double *cinputs = inputs + setindex * inputlayer->size;
            double *coutputs = outputs + setindex * outputlayer->size;

            network_forwardpass(network, cinputs);
            network_backwardpass(network, coutputs, learningrate);

        }

    }

}

static void xor_train(struct network *network, double epocs, double learningrate)
{

    double inputs[8] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    double outputs[4] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };
    unsigned int order[4];
    unsigned int i;

    for (i = 0; i < 4; i++)
        order[i] = i;

    train(network, epocs, learningrate, order, inputs, outputs, 4);

}

static void xor_validate1(struct network *network, double *inputs, double *outputs)
{

    struct nodelayer *last = network_getnodelayer(network, network->nsize - 1);
    unsigned int i;

    network_forwardpass(network, inputs);

    printf("Validating test:\n");

    for (i = 0; i < last->size; i++)
    {

        struct node *node = nodelayer_getnode(last, i);
        double output = outputs[i];

        printf("  [%u] Expected result %f\n", i, output);
        printf("  [%u] Actual result %f\n", i, node->output);

        if (fabs(node->output - output) < 0.5)
            printf("  [%u] Prediction: Correct\n", i);
        else
            printf("  [%u] Prediction: Wrong\n", i);

    }

}

static void xor_validate(struct network *network)
{

    static double inputs1[2] = {
        0.0f, 0.0f
    };
    static double inputs2[2] = {
        1.0f, 1.0f
    };
    static double inputs3[2] = {
        1.0f, 0.0f
    };
    static double inputs4[2] = {
        0.0f, 1.0f
    };
    static double outputs1[1] = {
        0.0f
    };
    static double outputs2[1] = {
        0.0f
    };
    static double outputs3[1] = {
        1.0f
    };
    static double outputs4[1] = {
        1.0f
    };

    xor_validate1(network, inputs1, outputs1);
    xor_validate1(network, inputs2, outputs2);
    xor_validate1(network, inputs3, outputs3);
    xor_validate1(network, inputs4, outputs4);

}

static void xor_run(void)
{

    struct nodelayer nodelayers[3];
    struct connectionlayer connectionlayers[2];
    struct network network;

    nodelayer_init(&nodelayers[0], 2);
    nodelayer_init(&nodelayers[1], 2);
    nodelayer_init(&nodelayers[2], 1);
    connectionlayer_init(&connectionlayers[0], &nodelayers[0], &nodelayers[1]);
    connectionlayer_init(&connectionlayers[1], &nodelayers[1], &nodelayers[2]);
    network_init(&network, nodelayers, 3, connectionlayers, 2);
    network_create(&network);
    xor_train(&network, 10000, 1.0f);
    xor_validate(&network);
    network_destroy(&network);

}

int main(int argc, const char **argv)
{

    xor_run();

    return 0;
}

