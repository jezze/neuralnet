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

struct connection *connectionlayer_getconnection(struct connectionlayer *clayer, unsigned int bindex, unsigned int aindex);
void connectionlayer_forwardpass(struct connectionlayer *layer);
void connectionlayer_backwardpass(struct connectionlayer *layer, double learningrate);
void connectionlayer_init(struct connectionlayer *layer, struct nodelayer *layerA, struct nodelayer *layerB);
void connectionlayer_create(struct connectionlayer *layer);
void connectionlayer_destroy(struct connectionlayer *layer);
