struct network
{

    struct nodelayer *nlayers;
    unsigned int nsize;
    struct connectionlayer *clayers;
    unsigned int csize;

};

struct nodelayer *network_getnodelayer(struct network *network, unsigned int index);
struct connectionlayer *network_getconnectionlayer(struct network *network, unsigned int index);
void network_forwardpass(struct network *network, double *inputs);
void network_backwardpass(struct network *network, double *outputs, double learningrate);
void network_init(struct network *network, struct nodelayer *nlayers, unsigned int nsize, struct connectionlayer *clayers, unsigned int csize);
void network_create(struct network *network);
void network_destroy(struct network *network);
