#include <stdlib.h>
#include <math.h>
#include "math.h"
#include "node.h"

struct node *nodelayer_getnode(struct nodelayer *layer, unsigned int index)
{

    return &layer->nodes[index];

}

static void node_init(struct node *node)
{

    node->output = 0.0f;
    node->delta = 0.0f;

}

void nodelayer_setinputs(struct nodelayer *layer, double *inputs)
{

    unsigned int i;

    for (i = 0; i < layer->size; i++)
    {

        struct node *node = &layer->nodes[i];
        double input = inputs[i];

        node->output = input;

    }

}

void nodelayer_setoutputs(struct nodelayer *layer, double *outputs)
{

    unsigned int i;

    for (i = 0; i < layer->size; i++)
    {

        struct node *node = nodelayer_getnode(layer, i);
        double output = outputs[i];

        node->delta = (output - node->output) * derived(node->output);

    }

}

void nodelayer_init(struct nodelayer *layer, unsigned int size)
{

    layer->nodes = 0;
    layer->size = size;

}

void nodelayer_create(struct nodelayer *layer)
{

    unsigned int i;

    layer->nodes = malloc(sizeof (struct node) * layer->size);

    for (i = 0; i < layer->size; i++)
        node_init(&layer->nodes[i]);

}

void nodelayer_destroy(struct nodelayer *layer)
{

    free(layer->nodes);

    layer->nodes = 0;

}

