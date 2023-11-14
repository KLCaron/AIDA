using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
        public MultinomialLogisticRegression(string fnProbabilities, string fnVocab, string fnMergedTfIdf, int numClasses)
        {
            _rand = new Random();
            List<Dictionary<string, object>> mergedTfIdf =
                ReadFile.ReadJson<List<Dictionary<string, Object>>>(fnMergedTfIdf);
            List<string> vocab = ReadFile.ReadJson<List<string>>(fnVocab);
            InitializeParameters(vocab, mergedTfIdf, numClasses);

            List<Dictionary<string, double>> allProbabilities = new List<Dictionary<string, double>>();

            foreach (var document in mergedTfIdf)
            {
                Dictionary<string, double> tfIdfScores = document["tfidf_scores"] as Dictionary<string, double>;
                Dictionary<string, double> classScores = ForwardPropagation(tfIdfScores, vocab);
                Dictionary<string, double> probabilities = SoftMax(classScores);

                allProbabilities.Add(probabilities);
            }

            File.WriteAllText(fnProbabilities, 
                JsonConvert.SerializeObject(allProbabilities, Formatting.Indented));
        }

        private void InitializeParameters(List<string> vocab, 
            List<Dictionary<string, object>> mergedTfIdf, int numClasses)
        {
            _numClasses = numClasses;
            _weights = InitializeWeights(vocab, mergedTfIdf, numClasses);
            _biases = InitializeBiases(_numClasses);
        }

        //also randomize weights, cannot start at 0
        private Dictionary<string, double> InitializeWeights(List<string> vocab, 
            List<Dictionary<string, Object>> mergedTfIdf, int numClasses)
        {
            Dictionary<string, double> initialWeights = new Dictionary<string, double>();

            foreach (var document in mergedTfIdf)
            {
                if (document["emotions"] is List<string> emotions)
                {
                    foreach (var term in vocab)
                    {
                        for (int classIndex = 0; classIndex < numClasses; classIndex++)
                        {
                            string weightKey = $"{term}_class{classIndex.ToString()}_emotion{emotions[0]}";
                            initialWeights[weightKey] = _rand.NextDouble() * 0.01;
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