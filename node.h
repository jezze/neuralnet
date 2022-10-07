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

struct node *nodelayer_getnode(struct nodelayer *layer, unsigned int index);
void nodelayer_setinputs(struct nodelayer *layer, double *inputs);
void nodelayer_setoutputs(struct nodelayer *layer, double *outputs);
void nodelayer_init(struct nodelayer *layer, unsigned int size);
void nodelayer_create(struct nodelayer *layer);
void nodelayer_destroy(struct nodelayer *layer);
