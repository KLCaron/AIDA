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
        private Dictionary<string, Dictionary<string, double>> _weights;
        private Dictionary<string, double> _biases;
        private readonly string[] _emotions = { "sadness", "joy", "love", "anger", "fear", "surprise" };
        private int _numClasses;
        private readonly Random _rand;

        //classes = emotions, 6 of em, sadness, joy, love, anger, fear, surprise
        //features is basically just the words in my vocab
        public MultinomialLogisticRegression(string fnProbabilities, string fnVocab, string fnTfIdf, int numClasses)
        {
            _rand = new Random();
            Dictionary<string, Dictionary<string, double>> tfIdf =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTfIdf);
            List<string> vocab = ReadFile.ReadJson<List<string>>(fnVocab);
            InitializeParameters(vocab, numClasses);
            
            //alright, I fucking give, I think I removed emotions downstream anyways, we're going back
            Dictionary<string, Dictionary<string, double>> classScores = ForwardPropagation(tfIdf);
            Dictionary<string, Dictionary<string, double>> probabilities = SoftMax(classScores);

            File.WriteAllText(fnProbabilities, 
                JsonConvert.SerializeObject(probabilities, Formatting.Indented));
        }

        private void InitializeParameters(List<string> vocab, int numClasses)
        {
            _numClasses = numClasses;
            _weights = InitializeWeights(vocab);
            _biases = InitializeBiases();
        }

        //also randomize weights, cannot start at 0
        private Dictionary<string, Dictionary<string, double>> InitializeWeights(List<string> vocab)
        {
            Dictionary<string, Dictionary<string, double>> initialWeights =
                new Dictionary<string, Dictionary<string, double>>();

            foreach (var emotion in _emotions)
            {
                initialWeights[emotion] = new Dictionary<string, double>();

                foreach (var term in vocab)
                {
                    initialWeights[emotion][term] = _rand.NextDouble() * 0.01;
                }
            }

            return initialWeights;
        }

        private Dictionary<string, double> InitializeBiases()
        {
            Dictionary<string, double> initialBiases = new Dictionary<string, double>();

            foreach (var emotion in _emotions)
            {
                initialBiases[emotion] = _rand.NextDouble() * 0.01;
            }

            return initialBiases;
        }

        //featureVector here is the tfidf score for 1 document/tweet
        //will need to iterate through the whole thing, so input vocab too
        //wherever I call this, will need to read my docs first
        //ok, so I need to go through each class, and each class's weight
        // my weights are string, string - double, so I can do foreach emotion there, then
        //for each string within there, and then I got to my tf-idf json, and if the emotion at that point matches
        //I then check the terms within, and multiply?
        private Dictionary<string,Dictionary<string, double>> ForwardPropagation(Dictionary<string, Dictionary<string, double>> tfIdf)
        {
            Dictionary<string, Dictionary<string, double>> classScores =
                new Dictionary<string, Dictionary<string, double>>();
            int i = 0;
            
            //all 6 emotions
            foreach (var emotion in _weights)
            {
                string emotionLabel = emotion.Key;
                Dictionary<string, double> emotionScores = new Dictionary<string, double>();
                double bias = _biases[emotionLabel];
                
                //every single term in vocab, once for all 6 emotions
                foreach (var weight in emotion.Value)
                {
                    string term = weight.Key;

                    //for every word under every emotion, we go through every tweet/document in our tfIdf
                    //top string is the emotion tied to it
                    foreach (var tfIdfEmotion in tfIdf)
                    {
                        string emotionKey = tfIdfEmotion.Key;
                        //we check if the emotion we're looking at is the right one
                        if (emotionKey.Split('_')[0] == emotionLabel.Split('_')[0])
                        {
                            //if it is, we iterate through every word-score pair in that tweet/document
                            foreach (var tfIdfTerm in tfIdfEmotion.Value)
                            {

                                string tfIdfTermKey = tfIdfTerm.Key;
                                //if the word we're looking at from weight is the same as the word we're looking at
                                //for tfIdf
                                if (tfIdfTermKey == term)
                                {
                                    double tfIdfValue = tfIdfTerm.Value;
                                    double score = weight.Value * tfIdfValue;

                                    //have we already looked at this word for this emotion? initialize if not
                                    if (!emotionScores.ContainsKey(term))
                                    {
                                        emotionScores[term] = score + bias;
                                    }
                                    else
                                    {
                                        emotionScores[term] += score;
                                    }
                                }
                            }
                        }
                    }
                }

                classScores[emotionLabel] = emotionScores;
            }

            return classScores;
        } 

        private Dictionary<string, Dictionary<string, double>> SoftMax(Dictionary<string, Dictionary<string, double>> classScores)
        {
            Dictionary<string, Dictionary<string, double>> probabilities =
                new Dictionary<string, Dictionary<string, double>>();

            var allEmotions = classScores.Keys.ToList();

            foreach (var term in classScores.SelectMany(emotion => emotion.Value.Keys).Distinct())
            {
                Dictionary<string, double> termProbabilities = new Dictionary<string, double>();

                foreach (var emotion in allEmotions)
                {
                    termProbabilities[emotion] = 0.0;
                }

                double maxScore =
                    classScores.Max(emotion => emotion.Value.TryGetValue(term, out var value) ? value : 0.0);
                double expSum = classScores.Sum(emotion =>
                    emotion.Value.TryGetValue(term, out var value1) ? Math.Exp(value1 - maxScore) : 0.0);

                foreach (var emotion in classScores)
                {
                    string emotionLabel = emotion.Key;
                    double expScore = emotion.Value.TryGetValue(term, out var value) ? 
                        Math.Exp(value - maxScore) : 0.0;
                    double probability = expScore / expSum;
                    termProbabilities[emotionLabel] = probability;
                }

                probabilities[term] = termProbabilities;
            }

            return probabilities;
            
        }
    }
}