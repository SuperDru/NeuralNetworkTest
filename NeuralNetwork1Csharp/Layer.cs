using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1Csharp
{
    class Layer
    {
        private Neuron[] neurons;

        public Neuron this[int i]
        {
            get { return neurons[i]; }
        }
        public int Size
        {
            get { return neurons.Length; }
        }

        public Layer(int numNeurons, int numOputputs)
        {
            neurons = new Neuron[numNeurons + 1];
            for (int i = 0; i <= numNeurons; i++)
                neurons[i] = new Neuron(numOputputs, i);
            neurons[numNeurons].OutputValue = 1f;
        }
    }
}
