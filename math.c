#include <stdlib.h>
#include <math.h>

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

