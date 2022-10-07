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

static void validatehelper(struct network *network, double *input, double *output)
{

    double confidence;
    double distance;

    network_forwardpass(network, input);

    confidence = network->nlayers[2].nodes[0].output;
    distance = fabs(confidence - output[0]);

    printf("Validating test:\n");
    printf("  Expected %f\n", output[0]);
    printf("  Confidence %f\n", confidence);

    if (distance < 0.5)
        printf("  Prediction: Correct\n");
    else
        printf("  Prediction: Wrong\n");

}

static void validate(struct network *network)
{

    static double input1[2] = {
        0.0f, 0.0f
    };
    static double input2[2] = {
        1.0f, 1.0f
    };
    static double input3[2] = {
        1.0f, 0.0f
    };
    static double input4[2] = {
        0.0f, 1.0f
    };
    static double output1[1] = {
        0.0f
    };
    static double output2[1] = {
        0.0f
    };
    static double output3[1] = {
        1.0f
    };
    static double output4[1] = {
        1.0f
    };

    validatehelper(network, input1, output1);
    validatehelper(network, input2, output2);
    validatehelper(network, input3, output3);
    validatehelper(network, input4, output4);

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

