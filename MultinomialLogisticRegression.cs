using System;
using System.Collections.Generic;

namespace AIDA
{
    //running with an object since it makes this more modular *just in case*
    public class MultinomialLogisticRegression
    {
        private Dictionary<string, double> _weights;
        private Dictionary<string, double> _biases;
        private readonly Random _rand;

        //classes = emotions, 6 of em, sadness, joy, love, anger, fear, surprise
        //features is basically just the words in my vocab
        public MultinomialLogisticRegression(string fnVocab, string fnTfIdf, int numClasses)
        {
            this._rand = new Random();
            InitializeParameters(fnVocab, fnTfIdf, numClasses);
        }

        private void InitializeParameters(string fnVocab, string fnTfIdf, int numClasses)
        {
            this._weights = InitializeWeights(fnVocab, fnTfIdf, numClasses);
            this._biases = InitializeBiases(numClasses);
        }

        private Dictionary<string, double> InitializeWeights(string fnVocab, string fnTfIdf, int numClasses)
        {
            Dictionary<string, Dictionary<string, double>> tfIdfScores =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTfIdf);
            List<string> vocab = ReadFile.ReadJson<List<string>>(fnVocab);
            Dictionary<string, double> initialWeights = new Dictionary<string, double>();

            foreach (var term in vocab)
            {
                foreach (var entry in tfIdfScores)
                {
                    string documentId = entry.Key;
                    Dictionary<string, double> tfIdfScoreSet = entry.Value;
                    
                    if (tfIdfScoreSet.TryGetValue(term, out double tfIdf))
                    {
                        for (int classIndex = 0; classIndex < numClasses; classIndex++)
                        {
                            string weightKey = $"{term}_class{classIndex.ToString()}_doc{documentId}";
                            initialWeights[weightKey] = tfIdf;
                        }
                    }
                }
            }

            return initialWeights;
        }

        private Dictionary<string, double> InitializeBiases(int numClasses)
        {
            Dictionary<string, double> initialBiases = new Dictionary<string, double>();

            for (int classIndex = 0; classIndex < numClasses; classIndex++)
            {
                initialBiases[$"class{classIndex.ToString()}"] = _rand.NextDouble() * 0.01;
            }

            return initialBiases;
        }
    }
}