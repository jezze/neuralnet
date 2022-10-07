#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "math.h"
#include "node.h"
#include "connection.h"
#include "network.h"

static struct connectionlayer *getconnectionlayer(struct network *network, unsigned int index)
{

    return &network->clayers[index];

}

struct nodelayer *network_getnodelayer(struct network *network, unsigned int index)
{

    return &network->nlayers[index];

}

void network_forwardpass(struct network *network, double *inputs)
{

    struct nodelayer *layer = network_getnodelayer(network, 0);
    unsigned int i;

    nodelayer_setinputs(layer, inputs);

    for (i = 0; i < network->csize; i++)
    {

        struct connectionlayer *layer = getconnectionlayer(network, i);

        connectionlayer_forwardpass(layer);

    }

}

void network_backwardpass(struct network *network, double *outputs, double learningrate)
{

    struct nodelayer *layer = network_getnodelayer(network, network->nsize - 1);
    unsigned int i;

    nodelayer_setoutputs(layer, outputs);

    for (i = network->csize; i > 0; i--)
    {

        struct connectionlayer *layer = getconnectionlayer(network, i - 1);

        connectionlayer_backwardpass(layer, learningrate);

    }

}

void network_init(struct network *network, struct nodelayer *nlayers, unsigned int nsize, struct connectionlayer *clayers, unsigned int csize)
{

    network->nlayers = nlayers;
    network->nsize = nsize;
    network->clayers = clayers;
    network->csize = csize;

}

void network_create(struct network *network)
{

    unsigned int i;

    for (i = 0; i < network->nsize; i++)
        nodelayer_create(&network->nlayers[i]);

    for (i = 0; i < network->csize; i++)
        connectionlayer_create(&network->clayers[i]);

}

void network_destroy(struct network *network)
{

    unsigned int i;

    for (i = 0; i < network->nsize; i++)
        nodelayer_destroy(&network->nlayers[i]);

    for (i = 0; i < network->csize; i++)
        connectionlayer_destroy(&network->clayers[i]);

}

