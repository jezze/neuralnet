#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NUMEPOCHS 10000
#define NUMNODELAYERS 3
#define NUMCONNECTIONLAYERS 2
#define LAYER0SIZE 2
#define LAYER1SIZE 2
#define LAYER2SIZE 1
#define NUMTRAININGSETS 4
#define LEARNINGRATE 0.1f

struct node
{

    double bias;
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
    unsigned int size;

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

double sigmoid_derived(double x)
{

    return x * (1 - x);

}

double randomize(void)
{

    return ((double)rand()) / ((double)RAND_MAX);

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

static void setinputs(struct nodelayer *layer, double *inputs)
{

    unsigned int j;

    for (j = 0; j < layer->size; j++)
    {

        struct node *node = &layer->nodes[j];

        node->output = inputs[j];

    }

}

static void forwardprop(struct nodelayer *layerA, struct nodelayer *layerB, struct connectionlayer *clayer)
{

    unsigned int j;

    for (j = 0; j < layerB->size; j++)
    {

        struct node *nodeB = &layerB->nodes[j];
        double activation = nodeB->bias;
        unsigned int k;

        for (k = 0; k < layerA->size; k++)
        {

            struct node *nodeA = &layerA->nodes[k];
            struct connection *connection = &clayer->connections[k * layerB->size + j];

            activation += nodeA->output * connection->weight;

        }

        nodeB->output = sigmoid(activation);

    }

}

static void setoutputs(struct nodelayer *layer, double *outputs)
{

    unsigned int j;

    for (j = 0; j < layer->size; j++)
    {

        struct node *node = &layer->nodes[j];
        double error = outputs[j] - node->output;

        node->delta = error * sigmoid_derived(node->output);

    }

}

static void backprop(struct nodelayer *layerA, struct nodelayer *layerB, struct connectionlayer *clayer)
{

    unsigned int j;

    for (j = 0; j < layerB->size; j++)
    {

        struct node *nodeB = &layerB->nodes[j];
        double error = 0.0f;
        unsigned int k;

        for (k = 0; k < layerA->size; k++)
        {

            struct node *nodeA = &layerA->nodes[k];
            struct connection *connection = &clayer->connections[j * layerA->size + k];

            error += nodeA->delta * connection->weight;

        }

        nodeB->delta = error * sigmoid_derived(nodeB->output);

    }

    for (j = 0; j < layerA->size; j++)
    {

        struct node *nodeA = &layerA->nodes[j];
        unsigned int k;

        nodeA->bias += nodeA->delta * LEARNINGRATE;

        for (k = 0; k < layerB->size; k++)
        {

            struct node *nodeB = &layerB->nodes[k];
            struct connection *connection = &clayer->connections[k * layerA->size + j];

            connection->weight += nodeB->output * nodeA->delta * LEARNINGRATE;

        }

    }

}

static void node_init(struct node *node)
{

    node->bias = randomize();
    node->output = 0.0f;
    node->delta = 0.0f;

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

static void connectionlayer_init(struct connectionlayer *layer, unsigned int size)
{

    layer->connections = 0;
    layer->size = size;

}

static void connectionlayer_create(struct connectionlayer *layer)
{

    unsigned int i;

    layer->connections = malloc(sizeof (struct connection) * layer->size);

    for (i = 0; i < layer->size; i++)
        connection_init(&layer->connections[i]);

}

static void connectionlayer_destroy(struct connectionlayer *layer)
{

    free(layer->connections);

    layer->connections = 0;

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

static void forwardpass(struct network *network, double *inputs)
{

    unsigned int n;

    setinputs(&network->nlayers[0], inputs);

    for (n = 0; n < network->nsize; n++)
        forwardprop(&network->nlayers[n], &network->nlayers[n + 1], &network->clayers[n]);

}

static void backwardpass(struct network *network, double *outputs)
{

    unsigned int n;

    setoutputs(&network->nlayers[network->nsize - 1], outputs);

    for (n = network->nsize - 1; n > 0; n--)
        backprop(&network->nlayers[n], &network->nlayers[n - 1], &network->clayers[n - 1]);

}

static void train(struct network *network)
{

    double training_inputs[NUMTRAININGSETS * LAYER0SIZE] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    double training_outputs[NUMTRAININGSETS * LAYER2SIZE] = {
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
            double *inputs = training_inputs + setindex * network->nlayers[0].size;
            double *outputs = training_outputs + setindex * network->nlayers[2].size;

            forwardpass(network, inputs);
            backwardpass(network, outputs);

        }

    }

}

static void validatehelper(struct network *network, double *input, double answer)
{

    double confidence;
    double distance;

    forwardpass(network, input);

    confidence = network->nlayers[2].nodes[0].output;
    distance = fabs(confidence - answer);

    printf("Validating test:\n");
    printf("  Confidence %f\n", confidence);

    if (distance < 0.5)
        printf("  Prediction: Correct\n");
    else
        printf("  Prediction: Wrong\n");

}

static void validate(struct network *network)
{

    static double input1[LAYER0SIZE] = {
        0.0f, 0.0f
    };
    static double input2[LAYER0SIZE] = {
        1.0f, 1.0f
    };
    static double input3[LAYER0SIZE] = {
        1.0f, 0.0f
    };
    static double input4[LAYER0SIZE] = {
        0.0f, 1.0f
    };

    validatehelper(network, input1, 0.0f);
    validatehelper(network, input2, 0.0f);
    validatehelper(network, input3, 1.0f);
    validatehelper(network, input4, 1.0f);

}

static struct nodelayer nodelayers[NUMNODELAYERS];
static struct connectionlayer connectionlayers[NUMCONNECTIONLAYERS];
static struct network network;

static void init(void)
{

    nodelayer_init(&nodelayers[0], LAYER0SIZE);
    nodelayer_init(&nodelayers[1], LAYER1SIZE);
    nodelayer_init(&nodelayers[2], LAYER2SIZE);
    connectionlayer_init(&connectionlayers[0], LAYER0SIZE * LAYER1SIZE);
    connectionlayer_init(&connectionlayers[1], LAYER1SIZE * LAYER2SIZE);
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
