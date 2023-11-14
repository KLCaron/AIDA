using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using Newtonsoft.Json;

namespace AIDA
{
    //running with an object since it makes this more modular *just in case*
    public class MultinomialLogisticRegression
    {
        private Dictionary<string, double> _weights;
        private Dictionary<string, double> _biases;
        private int _numClasses;
        private readonly Random _rand;

        //classes = emotions, 6 of em, sadness, joy, love, anger, fear, surprise
        //features is basically just the words in my vocab
        public MultinomialLogisticRegression(string fnProbabilities, string fnVocab, string fnTfIdf, int numClasses)
        {
            _rand = new Random();
            Dictionary<string, Dictionary<string, double>> tfIdfScores =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTfIdf);
            List<string> vocab = ReadFile.ReadJson<List<string>>(fnVocab);
            InitializeParameters(vocab, tfIdfScores, numClasses);

            List<Dictionary<string, double>> allProbabilities = new List<Dictionary<string, double>>();

            foreach (var featureVector in tfIdfScores.Values)
            {
                Dictionary<string, double> classScores = ForwardPropagation(featureVector, vocab);
                Dictionary<string, double> probabilities = SoftMax(classScores);

                allProbabilities.Add(probabilities);
            }

            File.WriteAllText(fnProbabilities, 
                JsonConvert.SerializeObject(allProbabilities, Formatting.Indented));
        }

        private void InitializeParameters(List<string> vocab, 
            Dictionary<string, Dictionary<string, double>> tfIdfScores, int numClasses)
        {
            _numClasses = numClasses;
            _weights = InitializeWeights(vocab, tfIdfScores, numClasses);
            _biases = InitializeBiases(_numClasses);
        }

        private Dictionary<string, double> InitializeWeights(List<string> vocab, 
            Dictionary<string, Dictionary<string, double>> tfIdfScores, int numClasses)
        {
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

        //featureVector here is the tfidf score for 1 document/tweet
        //will need to iterate through the whole thing, so input vocab too
        //wherever I call this, will need to read my docs first
        private Dictionary<string, double> ForwardPropagation(Dictionary<string, double> featureVector, 
            List<string> vocab)
        {
            Dictionary<string, double> classScores = new Dictionary<string, double>();

            foreach (var entry in _biases)
            {
                string classLabel = entry.Key;
                double bias = entry.Value;
                double score = bias;

                foreach (var term in vocab)
                {
                    if (featureVector.TryGetValue(term, out double tfIdf))
                    {
                        string weightKey = $"{term}_class{classLabel}";
                        if (_weights.TryGetValue(weightKey, out double weight))
                        {
                            score += tfIdf * weight;
                        }
                    }
                }

                classScores[classLabel] = score;
            }

            return classScores;
        }

        private Dictionary<string, double> SoftMax(Dictionary<string, double> classScores)
        {
            double maxScore = classScores.Max(x => x.Value);

            //subtracting math score for numerical stability?
            double expSum = classScores.Values.Sum(x => Math.Exp(x - maxScore));

            Dictionary<string, double> probabilities = new Dictionary<string, double>();

            foreach (var entry in classScores)
            {
                string classLabel = entry.Key;
                double score = entry.Value;

                double expScore = Math.Exp(score - maxScore);
                double probability = expScore / expSum;
                probabilities[classLabel] = probability;
            }

            return probabilities;
        }
    }
}