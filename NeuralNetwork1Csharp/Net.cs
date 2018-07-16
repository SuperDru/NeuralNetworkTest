using System;

namespace NeuralNetwork1Csharp
{
    class Net
    {
        float eta = 0.15f;
        private int numLayers;
        private Layer[] layers;

        public Net(int[] topology)
        {
            numLayers = topology.Length;
            layers = new Layer[numLayers];
            for (int i = 0; i < numLayers - 1; i++)
                layers[i] = new Layer(topology[i], topology[i + 1]);
            layers[numLayers - 1] = new Layer(topology[numLayers - 1], 0);
        }

        public void feedForward(float[] inputValues)
        {
            if (inputValues.Length != layers[0].Size - 1)
                throw new Exception("The number of input values doesn't match the network topology");

            for (int i = 0; i < layers[0].Size - 1; i++)
                layers[0][i].OutputValue = inputValues[i];

            for (int i = 1; i < numLayers; i++)
                for (int j = 0; j < layers[i].Size - 1; j++)
                    layers[i][j].feedForward(layers[i - 1]);
        }

        public void backPropagation(float[] targetValues, int max)
        {
            if (targetValues.Length != layers[numLayers - 1].Size - 1)
                throw new Exception("The number of target values doesn't match the network topology");

            for (int i = 0; i < targetValues.Length; i++)
                targetValues[i] = 1 / targetValues[i];

            Layer layer = layers[numLayers - 1];
            Layer prevLayer = layers[numLayers - 2];
            for (int i = 0; i < targetValues.Length; i++)
            {
                float d = targetValues[i] - layer[i].OutputValue;
                float derivative = layer[i].getDerivative();
                layer[i].Sigma = d * derivative;
                
                for (int j = 0; j < layers[numLayers - 2].Size; j++)
                {
                    float delta = eta * d * derivative * prevLayer[j].OutputValue;
                    prevLayer[j].Sigma += d * derivative * prevLayer[j][i];
                    prevLayer[j][i] += delta;
                }
            }

            for (int i = numLayers - 2; i > 0; i--)
            {
                layer = layers[i];
                prevLayer = layers[i - 1];
                for (int j = 0; j < layer.Size; j++)
                {
                    float derivative = layer[j].getDerivative();

                    layer[j].Sigma *= derivative;

                    for (int k = 0; k < prevLayer.Size; k++)
                    {
                        float delta = eta * layer[j].Sigma * prevLayer[k].OutputValue;
                        prevLayer[k].Sigma += layer[j].Sigma * prevLayer[k][j];
                        prevLayer[k][j] += delta;
                    }
                }

            }
        }

        public float[] getResults()
        {
            float[] result = new float[layers[numLayers - 1].Size - 1];
            for (int i = 0; i < result.Length; i++)
                result[i] = 1 / layers[numLayers - 1][i].OutputValue;
            return result;
        }
    }
}
