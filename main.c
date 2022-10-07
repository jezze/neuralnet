#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "math.h"
#include "node.h"
#include "connection.h"
#include "network.h"

#define NUMEPOCHS 10000
#define NUMNODELAYERS 3
#define NUMCONNECTIONLAYERS 2
#define NUMTRAININGSETS 4
#define LEARNINGRATE 1.0f

static struct nodelayer nodelayers[NUMNODELAYERS];
static struct connectionlayer connectionlayers[NUMCONNECTIONLAYERS];
static struct network network;

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

static void train(struct network *network)
{

    struct nodelayer *inputlayer = network_getnodelayer(network, 0);
    struct nodelayer *outputlayer = network_getnodelayer(network, network->nsize - 1);

    double training_inputs[NUMTRAININGSETS * 2] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    double training_outputs[NUMTRAININGSETS] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };
    unsigned int training_order[NUMTRAININGSETS];
    unsigned int epoch;
    unsigned int i;

    for (i = 0; i < NUMTRAININGSETS; i++)
        training_order[i] = i;

    for (epoch = 0; epoch < NUMEPOCHS; epoch++)
    {

        unsigned int i;

        shuffle(training_order, NUMTRAININGSETS);

        for (i = 0; i < NUMTRAININGSETS; i++)
        {

            unsigned int setindex = training_order[i];
            double *inputs = training_inputs + setindex * inputlayer->size;
            double *outputs = training_outputs + setindex * outputlayer->size;

            network_forwardpass(network, inputs);
            network_backwardpass(network, outputs, LEARNINGRATE);

        }

    }

}

static void validate1(struct network *network, double *inputs, double *outputs)
{

    struct nodelayer *last = network_getnodelayer(network, network->nsize - 1);
    double confidence;
    double distance;
    unsigned int i;

    network_forwardpass(network, inputs);

    printf("Validating test:\n");

    for (i = 0; i < last->size; i++)
    {

        struct node *node = nodelayer_getnode(last, i);
        double output = outputs[i];

        confidence = node->output;
        distance = fabs(confidence - output);

        printf("  [%u] Expected %f\n", i, output);
        printf("  [%u] Confidence %f\n", i, confidence);

        if (distance < 0.5)
            printf("  [%u] Prediction: Correct\n", i);
        else
            printf("  [%u] Prediction: Wrong\n", i);

    }

}

static void validate(struct network *network)
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

    validate1(network, inputs1, outputs1);
    validate1(network, inputs2, outputs2);
    validate1(network, inputs3, outputs3);
    validate1(network, inputs4, outputs4);

}

static void init(void)
{

    nodelayer_init(&nodelayers[0], 2);
    nodelayer_init(&nodelayers[1], 2);
    nodelayer_init(&nodelayers[2], 1);
    connectionlayer_init(&connectionlayers[0], &nodelayers[0], &nodelayers[1]);
    connectionlayer_init(&connectionlayers[1], &nodelayers[1], &nodelayers[2]);
    network_init(&network, nodelayers, NUMNODELAYERS, connectionlayers, NUMCONNECTIONLAYERS);

}

int main(int argc, const char **argv)
{

    init();
    network_create(&network);
    train(&network);
    validate(&network);
    network_destroy(&network);

    return 0;
}

