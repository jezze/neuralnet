#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NUMEPOCHS 10000
#define NUMINPUTS 2
#define NUMHIDDENNODES 2
#define NUMOUTPUTS 1
#define NUMTRAININGSETS 4

double sigmoid(double x)
{

    return 1 / (1 + exp(-x));

}

double dSigmoid(double x)
{

    return x * (1 - x);

}

double init_weight()
{

    return ((double)rand()) / ((double)RAND_MAX);

}

void shuffle(int *array, unsigned int n)
{

    unsigned int i;

    for (i = 0; i < n - 1; i++)
    {

        unsigned int j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];

        array[j] = array[i];
        array[i] = t;

    }

}

int main(int argc, const char **argv)
{

    const double lr = 0.1f;
    int trainingSetOrder[NUMTRAININGSETS] = {0, 1, 2, 3};
    double hiddenLayer[NUMHIDDENNODES];
    double outputLayer[NUMOUTPUTS];
    double hiddenLayerBias[NUMHIDDENNODES];
    double outputLayerBias[NUMOUTPUTS];
    double hiddenWeights[NUMINPUTS][NUMHIDDENNODES];
    double outputWeights[NUMHIDDENNODES][NUMOUTPUTS];
    double training_inputs[NUMTRAININGSETS][NUMINPUTS] = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };
    double training_outputs[NUMTRAININGSETS][NUMOUTPUTS] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    for (int i = 0; i < NUMINPUTS; i++)
        for (int j = 0; j < NUMHIDDENNODES; j++)
            hiddenWeights[i][j] = init_weight();

    for (int i = 0; i < NUMHIDDENNODES; i++)
    {

        hiddenLayerBias[i] = init_weight();

        for (int j = 0; j < NUMOUTPUTS; j++)
            outputWeights[i][j] = init_weight();

    }

    for (int i = 0; i < NUMOUTPUTS; i++)
        outputLayerBias[i] = init_weight();

    for (int n = 0; n < NUMEPOCHS; n++)
    {

        shuffle(trainingSetOrder, NUMTRAININGSETS);

        for (int x = 0; x < NUMTRAININGSETS; x++)
        {

            int i = trainingSetOrder[x];

            // Forward pass

            for (int j = 0; j < NUMHIDDENNODES; j++)
            {

                double activation = hiddenLayerBias[j];

                for (int k = 0; k < NUMINPUTS; k++)
                    activation += training_inputs[i][k] * hiddenWeights[k][j];

                hiddenLayer[j] = sigmoid(activation);

            }

            for (int j = 0; j < NUMOUTPUTS; j++)
            {

                double activation=outputLayerBias[j];

                for (int k = 0; k < NUMHIDDENNODES; k++)
                    activation += hiddenLayer[k] * outputWeights[k][j];

                outputLayer[j] = sigmoid(activation);

            }

            printf("Input\n");
            printf("  %f\n", training_inputs[i][0]);
            printf("  %f\n", training_inputs[i][1]);
            printf("Output\n");
            printf("  %f\n", outputLayer[0]);
            printf("Expected output\n");
            printf("  %f\n", training_outputs[i][0]);

            // Backprop

            double deltaOutput[NUMOUTPUTS];

            for (int j = 0; j < NUMOUTPUTS; j++)
            {

                double errorOutput = training_outputs[i][j] - outputLayer[j];

                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);

            }

            double deltaHidden[NUMHIDDENNODES];

            for (int j = 0; j < NUMHIDDENNODES; j++)
            {

                double errorHidden = 0.0f;

                for (int k = 0; k < NUMOUTPUTS; k++)
                    errorHidden += deltaOutput[k] * outputWeights[j][k];

                deltaHidden[j] = errorHidden*dSigmoid(hiddenLayer[j]);

            }

            for (int j = 0; j < NUMOUTPUTS; j++)
            {

                outputLayerBias[j] += deltaOutput[j] * lr;

                for (int k = 0; k < NUMHIDDENNODES; k++)
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;

            }

            for (int j = 0; j < NUMHIDDENNODES; j++)
            {

                hiddenLayerBias[j] += deltaHidden[j] * lr;

                for (int k = 0; k < NUMINPUTS; k++)
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;

            }

        }

    }

    printf("Final Hidden Weights\n");
    printf("[\n");

    for (int j = 0; j < NUMHIDDENNODES; j++)
    {

        for (int k = 0; k < NUMINPUTS; k++)
            printf("  %f\n", hiddenWeights[k][j]);

    }

    printf("]\n");
    printf("Final Hidden Biases\n");
    printf("[\n");

    for (int j = 0; j < NUMHIDDENNODES; j++)
        printf("  %f\n", hiddenLayerBias[j]);

    printf("]\n");
    printf("Final Output Weights\n");
    printf("[\n");

    for (int j = 0; j < NUMOUTPUTS; j++)
    {

        for (int k = 0; k < NUMHIDDENNODES; k++)
            printf("  %f\n", outputWeights[k][j]);

    }

    printf("]\n");
    printf("Final Output Biases\n");
    printf("[\n");

    for (int j = 0; j < NUMOUTPUTS; j++)
        printf("  %f\n", outputLayerBias[j]);

    printf("]\n");

    return 0;
}

