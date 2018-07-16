using System;

namespace NeuralNetwork1Csharp
{
    class Neuron
    {
        static private Random rand;
        private float[] weights;
        private float outputValue;
        private float sigma = 0f;
        private int number;

        public float this[int i]
        {
            get { return weights[i]; }
            set { weights[i] = value; }
        }
        public float Sigma
        {
            get { return sigma; }
            set { sigma = value; }
        }
        public float OutputValue
        {
            get { return outputValue; }
            set { outputValue = value; }
        }

        public Neuron(int numOutputs, int number)
        {
            this.number = number;
            rand = new Random();
            weights = new float[numOutputs + 1];
            for (int i = 0; i < numOutputs + 1; i++)
                weights[i] = (float)rand.NextDouble();
        }


        public void feedForward(Layer prevLayer)
        {
            float sum = 0f;
            for (int i = 0; i < prevLayer.Size; i++)
                sum += prevLayer[i].OutputValue * prevLayer[i].weights[number];
            outputValue = activation(sum);
        }

        public float activation(float sum)
        {
            return (float)(1 / (1 + Math.Exp(-sum)));
        }
        public float getDerivative()
        {
            return outputValue * (1 - outputValue);
        }
    }
}
