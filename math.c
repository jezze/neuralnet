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

void shuffle(unsigned int *a, unsigned int n)
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

