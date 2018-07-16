using System;
using System.Text;
using System.IO;

namespace NeuralNetwork1Csharp
{
    class Program
    {
        static void Main(string[] args)
        {
            //StreamWriter sw = new StreamWriter("data.txt");
            //Random rand = new Random();

            //for (int i = 0; i < 100000000; i++)
            //{
            //    int a = rand.Next(1, 10);
            //    int b = rand.Next(1, 10);
            //    sw.WriteLine(string.Format("a = {0}, b = {1}", a, b));
            //    sw.WriteLine(string.Format("a * b = {0}", a * b));
            //}
            //sw.Close();


            Net net = new Net(new int[] { 2, 15, 10, 5, 1 });

            StreamReader sr = new StreamReader("data.txt");

            while (!sr.EndOfStream)
            {
                string line = sr.ReadLine();
                float[] inputValues = new float[2];
                inputValues[0] = Convert.ToInt32(line[4].ToString());
                inputValues[1] = Convert.ToInt32(line[11].ToString());

                Console.WriteLine(string.Format("a = {0}, b = {1}", inputValues[0], inputValues[1]));

                line = sr.ReadLine();
                float[] targerValues = new float[1];
                targerValues[0] = Convert.ToInt32(line.Split(' ')[4]);

                Console.WriteLine(string.Format("target value = {0}", targerValues[0]));

                net.feedForward(inputValues);

                double resultValue = Math.Round(net.getResults()[0]);

                Console.WriteLine(string.Format("result value = {0}", resultValue));

                net.backPropagation(targerValues, 100);
            }
            sr.Close();

            Console.ReadKey();
        }
    }
}
