#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct node
{

    double output;
    double delta;

};

struct nodelayer
{

    struct node *nodes;
    unsigned int size;

};

struct connection
{

    double weight;

};

struct connectionlayer
{

    struct connection *connections;
    struct nodelayer *nlayerA;
    struct nodelayer *nlayerB;

};

struct network
{

    struct nodelayer *nlayers;
    unsigned int nsize;
    struct connectionlayer *clayers;
    unsigned int csize;

};

double sigmoid(double x)
{

    return 1 / (1 + exp(-x));

}

double derived(double x)
{

    return x * (1 - x);

}

double randomize(void)
{

    return ((double)rand()) / ((double)RAND_MAX);

}

static struct nodelayer *network_getnodelayer(struct network *network, unsigned int index)
{

    return &network->nlayers[index];

}

static struct connectionlayer *network_getconnectionlayer(struct network *network, unsigned int index)
{

    return &network->clayers[index];

}

static struct node *nodelayer_getnode(struct nodelayer *layer, unsigned int index)
{

    return &layer->nodes[index];

}

static struct connection *connectionlayer_getconnection(struct connectionlayer *clayer, unsigned int bindex, unsigned int aindex)
{

    return &clayer->connections[bindex * clayer->nlayerB->size + aindex];

}

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

static void node_init(struct node *node)
{

    node->output = 0.0f;
    node->delta = 0.0f;

}

static void nodelayer_setinputs(struct nodelayer *layer, double *inputs)
{

    unsigned int i;

    for (i = 0; i < layer->size; i++)
    {

        struct node *node = &layer->nodes[i];

        node->output = inputs[i];

    }

}

static void nodelayer_setoutputs(struct nodelayer *layer, double *outputs)
{

    unsigned int i;

    for (i = 0; i < layer->size; i++)
    {

        struct node *node = nodelayer_getnode(layer, i);
        double error = outputs[i] - node->output;

        node->delta = error * derived(node->output);

    }

}

static void nodelayer_init(struct nodelayer *layer, unsigned int size)
{

    layer->nodes = 0;
    layer->size = size;

}

static void nodelayer_create(struct nodelayer *layer)
{

    unsigned int i;

    layer->nodes = malloc(sizeof (struct node) * layer->size);

    for (i = 0; i < layer->size; i++)
        node_init(&layer->nodes[i]);

}

static void nodelayer_destroy(struct nodelayer *layer)
{

    free(layer->nodes);

    layer->nodes = 0;

}

static void connection_init(struct connection *connection)
{

    connection->weight = randomize();

}

static void connectionlayer_forwardpass(struct connectionlayer *layer)
{

    unsigned int a;
    unsigned int b;

    for (b = 0; b < layer->nlayerB->size; b++)
    {

        struct node *nodeB = nodelayer_getnode(layer->nlayerB, b);
        double activation = 0.0f;

        for (a = 0; a < layer->nlayerA->size; a++)
        {

            struct node *nodeA = nodelayer_getnode(layer->nlayerA, a);
            struct connection *connection = connectionlayer_getconnection(layer, a, b);

            activation += nodeA->output * connection->weight;

        }

        nodeB->output = sigmoid(activation);

    }

}

static void connectionlayer_backwardpass(struct connectionlayer *layer, double learningrate)
{

    unsigned int a;
    unsigned int b;

    for (a = 0; a < layer->nlayerA->size; a++)
    {

        struct node *nodeA = nodelayer_getnode(layer->nlayerA, a);
        double error = 0.0f;

        for (b = 0; b < layer->nlayerB->size; b++)
        {

            struct node *nodeB = nodelayer_getnode(layer->nlayerB, b);
            struct connection *connection = connectionlayer_getconnection(layer, a, b);

            error += nodeB->delta * connection->weight;

        }

        nodeA->delta = error * derived(nodeA->output);

    }

    for (a = 0; a < layer->nlayerA->size; a++)
    {

        struct node *nodeA = nodelayer_getnode(layer->nlayerA, a);

        for (b = 0; b < layer->nlayerB->size; b++)
        {

            struct node *nodeB = nodelayer_getnode(layer->nlayerB, b);
            struct connection *connection = connectionlayer_getconnection(layer, a, b);

            connection->weight += nodeA->output * nodeB->delta * learningrate;

        }

    }

}

static void connectionlayer_init(struct connectionlayer *layer, struct nodelayer *layerA, struct nodelayer *layerB)
{

    layer->connections = 0;
    layer->nlayerA = layerA;
    layer->nlayerB = layerB;

}

static void connectionlayer_create(struct connectionlayer *layer)
{

    unsigned int size = layer->nlayerA->size * layer->nlayerB->size;
    unsigned int i;

    layer->connections = malloc(sizeof (struct connection) * size);

    for (i = 0; i < size; i++)
        connection_init(&layer->connections[i]);

}

static void connectionlayer_destroy(struct connectionlayer *layer)
{

    free(layer->connections);

    layer->connections = 0;

}

static void network_forwardpass(struct network *network, double *inputs)
{

    struct nodelayer *layer = network_getnodelayer(network, 0);
    unsigned int i;

    nodelayer_setinputs(layer, inputs);

    for (i = 0; i < network->csize; i++)
    {

        struct connectionlayer *layer = network_getconnectionlayer(network, i);

        connectionlayer_forwardpass(layer);

    }

}

static void network_backwardpass(struct network *network, double *outputs, double learningrate)
{

    struct nodelayer *layer = network_getnodelayer(network, network->nsize - 1);
    unsigned int i;

    nodelayer_setoutputs(layer, outputs);

    for (i = network->csize; i > 0; i--)
    {

        struct connectionlayer *layer = network_getconnectionlayer(network, i - 1);

        connectionlayer_backwardpass(layer, learningrate);

    }

}

static void network_init(struct network *network, struct nodelayer *nlayers, unsigned int nsize, struct connectionlayer *clayers, unsigned int csize)
{

    network->nlayers = nlayers;
    network->nsize = nsize;
    network->clayers = clayers;
    network->csize = csize;

}

static void network_create(struct network *network)
{

    unsigned int i;

    for (i = 0; i < network->nsize; i++)
        nodelayer_create(&network->nlayers[i]);

    for (i = 0; i < network->csize; i++)
        connectionlayer_create(&network->clayers[i]);

}

static void network_destroy(struct network *network)
{

    unsigned int i;

    for (i = 0; i < network->nsize; i++)
        nodelayer_destroy(&network->nlayers[i]);

    for (i = 0; i < network->csize; i++)
        connectionlayer_destroy(&network->clayers[i]);

}

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

