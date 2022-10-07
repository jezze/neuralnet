#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "math.h"
#include "node.h"
#include "connection.h"

struct connection *connectionlayer_getconnection(struct connectionlayer *clayer, unsigned int bindex, unsigned int aindex)
{

    return &clayer->connections[bindex * clayer->nlayerB->size + aindex];

}

static void connection_init(struct connection *connection)
{

    connection->weight = randomize();

}

void connectionlayer_forwardpass(struct connectionlayer *layer)
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

void connectionlayer_backwardpass(struct connectionlayer *layer, double learningrate)
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

void connectionlayer_init(struct connectionlayer *layer, struct nodelayer *layerA, struct nodelayer *layerB)
{

    layer->connections = 0;
    layer->nlayerA = layerA;
    layer->nlayerB = layerB;

}

void connectionlayer_create(struct connectionlayer *layer)
{

    unsigned int size = layer->nlayerA->size * layer->nlayerB->size;
    unsigned int i;

    layer->connections = malloc(sizeof (struct connection) * size);

    for (i = 0; i < size; i++)
        connection_init(&layer->connections[i]);

}

void connectionlayer_destroy(struct connectionlayer *layer)
{

    free(layer->connections);

    layer->connections = 0;

}

