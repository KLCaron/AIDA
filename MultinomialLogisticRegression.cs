using System;
using System.Collections.Generic;
using System.Globalization;
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
        private readonly Random _rand;
        
        /*
         * initialize a starting model object
         */
        public MultinomialLogisticRegression(string fnVocab, string fnMlr)
        {
            _rand = new Random();
            List<string> vocab = ReadFile.ReadJson<List<string>>(fnVocab);
            InitializeParameters(vocab);
            
            SaveModel(fnMlr);
        }

        /*
         * save the model object, to re-use later
         */
        private void SaveModel(string fnMlr)
        {
            var modelData = new
            {
                Weights = _weights,
                Biases = _biases,
                RandSeed = _rand?.Next(0, int.MaxValue)
            };

            string jsonData = JsonConvert.SerializeObject(modelData, Formatting.Indented);
            File.WriteAllText(fnMlr, jsonData);
        }

        /*
         * to perform all the actual work on the model
         */
        public MultinomialLogisticRegression(int choice, string fnMlr, string fnTfIdf, string fnProbabilities, 
            string fnMergedProbabilities, string fnCorpus, string fnDocuments, string fnAggregatedProbabilities, 
            string fnLossSet, string fnAverageLoss, string fnVocab, string fnTermLossSet, double learningRate)
        {
            string jsonData = File.ReadAllText(fnMlr);
            var modelData = JsonConvert.DeserializeObject<dynamic>(jsonData);

            _weights = modelData.Weights.ToObject<Dictionary<string, Dictionary<string, double>>>();
            _biases = modelData.Biases.ToObject<Dictionary<string, double>>();

            int? randSeed = modelData.RandSeed;
            _rand = randSeed.HasValue ? new Random(randSeed.Value) : new Random();

            Dictionary<int, Action> actions = new Dictionary<int, Action>
            {
                {0, () => MlrForwardPropSoftMax(fnProbabilities, fnTfIdf)},
                {1, () => MergeDocumentsTermProbabilities(fnMergedProbabilities, fnCorpus, fnDocuments,
                    fnProbabilities)},
                {2, () => AggregateDocumentProbabilities(fnAggregatedProbabilities, fnMergedProbabilities)},
                {3, () => CalcCrossEntropyLoss(fnAggregatedProbabilities, fnLossSet, fnAverageLoss)},
                {4, () => DocumentLossToTermLoss(fnLossSet, fnTfIdf, fnVocab, fnTermLossSet)},
                {5, () => GradientDescent(fnTermLossSet, learningRate)}
            };

            if (actions.TryGetValue(choice, out Action action))
            {
                action();
            }

            SaveModel(fnMlr);
        }

        private void MlrForwardPropSoftMax(string fnProbabilities, string fnTfIdf)
        {
            Dictionary<string, Dictionary<string, double>> tfIdf =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTfIdf);
            
            Dictionary<string, Dictionary<string, double>> classScores = ForwardPropagation(tfIdf);
            Dictionary<string, Dictionary<string, double>> probabilities = SoftMax(classScores);

            File.WriteAllText(fnProbabilities, 
                JsonConvert.SerializeObject(probabilities, Formatting.Indented));
        }

        private void InitializeParameters(List<string> vocab)
        {
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
        private Dictionary<string,Dictionary<string, double>> ForwardPropagation(Dictionary<string, 
            Dictionary<string, double>> tfIdf)
        {
            Dictionary<string, Dictionary<string, double>> classScores =
                new Dictionary<string, Dictionary<string, double>>();

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

        private Dictionary<string, Dictionary<string, double>> SoftMax(Dictionary<string, 
            Dictionary<string, double>> classScores)
        {
            Dictionary<string, Dictionary<string, double>> probabilities =
                new Dictionary<string, Dictionary<string, double>>();

            var allEmotions = classScores.Keys.ToList();

            foreach (var term in classScores.SelectMany(emotion => 
                         emotion.Value.Keys).Distinct())
            {
                Dictionary<string, double> termProbabilities = new Dictionary<string, double>();

                foreach (var emotion in allEmotions)
                {
                    termProbabilities[emotion] = 0.0;
                }

                double maxScore =
                    classScores.Max(emotion => emotion.Value.TryGetValue(term, 
                        out var value) ? value : 0.0);
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

        /*
         * merge probabilities are the document, its true emotion, and then each term inside with probabilities
         * applied at the term level
         */
        private void MergeDocumentsTermProbabilities(string fnMergedProbabilities, string fnCorpus, string fnDocuments,
            string fnAllTermProbabilities)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);
            List<Dictionary<string, string>> documents =
                ReadFile.ReadJson<List<Dictionary<string, string>>>(fnDocuments);
            Dictionary<string, Dictionary<string, double>> allTermProbabilities =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnAllTermProbabilities);
            Dictionary<string, Dictionary<string, Dictionary<string, double>>> mergedDocuments =
                new Dictionary<string, Dictionary<string, Dictionary<string, double>>>();
            int i = 0;
            
            foreach (var document in documents)
            {
                
                string text = string.Join(" ", corpus[i]);
                string trueEmotion = document["emotions"];

                Dictionary<string, Dictionary<string, double>> documentProbabilities =
                    new Dictionary<string, Dictionary<string, double>>();

                documentProbabilities[trueEmotion] = null;

                foreach (var term in corpus[i])
                {
                    if (allTermProbabilities.TryGetValue(term, out var probability))
                    {
                        documentProbabilities[term] = probability;
                    }
                }

                mergedDocuments[text] = documentProbabilities;
                i++;
            }

            File.WriteAllText(fnMergedProbabilities, 
                JsonConvert.SerializeObject(mergedDocuments, Formatting.Indented));
        }

        /*
         * aggregate probabilities are the document, with the true emotion, and then probabilities for each
         * emotion.
         */
        private void AggregateDocumentProbabilities(string fnAggregatedProbabilities, string fnMergedProbabilities)
        {
            Dictionary<string, Dictionary<string, double>> aggregatedProbabilities =
                new Dictionary<string, Dictionary<string, double>>();
            Dictionary<string, Dictionary<string, Dictionary<string, double>>> mergedProbabilities =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, Dictionary<string, double>>>>(
                    fnMergedProbabilities);

            foreach (var document in mergedProbabilities)
            {
                string documentKey = document.Key;
                string trueEmotionKey = "_" + document.Value.Keys.First();
                Dictionary<string, double> emotionSet = new Dictionary<string, double>();
                emotionSet[trueEmotionKey] = 0.0;

                foreach (var emotion in _emotions)
                {
                    emotionSet[emotion] = 0.0;
                }

                bool firstTermProbabilities = true;
                
                foreach (var termProbabilities in document.Value)
                {
                    if (!firstTermProbabilities)
                    {
                        foreach (var emotion in termProbabilities.Value)
                        {
                            emotionSet[emotion.Key] += emotion.Value;
                        }                        
                    }
                    else
                    {
                        firstTermProbabilities = false;
                    }

                }

                double sum = emotionSet.Sum(kv => kv.Value);
                foreach (var emotion in _emotions)
                {
                    emotionSet[emotion] /= sum;
                }
                aggregatedProbabilities[documentKey] = emotionSet;
            }
            
            File.WriteAllText(fnAggregatedProbabilities, 
                JsonConvert.SerializeObject(aggregatedProbabilities, Formatting.Indented));
        }
        
        /*
         * calculates loss score for each document, underneath each emotion. Set to 0 by default.
         */
        private void CalcCrossEntropyLoss(string fnAggregatedProbabilities, string fnLossSet, string fnAverageLoss)
        {
            Dictionary<string, Dictionary<string, double>> aggregatedProbabilities =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnAggregatedProbabilities);
            //Dictionary<string, double> lossSet = new Dictionary<string, double>();
            Dictionary<string, Dictionary<string, double>> lossSet =
                new Dictionary<string, Dictionary<string, double>>();
            double totalLoss = 0.0;

            foreach (var emotion in _emotions)
            {
                lossSet[emotion] = new Dictionary<string, double>();
            }
            
            foreach (var document in aggregatedProbabilities)
            {
                double totalLossForDocument = 0.0;
                string documentKey = document.Key;
                string trueEmotion = document.Value.Keys.First().Substring(1);
                var documentVal = document.Value;
                
                foreach (var emotion in _emotions)
                {
                    double indicator = (emotion == trueEmotion) ? 1.0 : 0.0;
                    double predictedProbability = documentVal[emotion];
                                
                    //need to avoid log(0)
                    double epsilon = 1e-15;
                    predictedProbability = Math.Max(epsilon, Math.Min(1 - epsilon, predictedProbability));

                    double loss = -indicator * Math.Log(predictedProbability);
                    totalLossForDocument += loss;

                    lossSet[emotion][documentKey] = loss;
                }

                totalLoss += totalLossForDocument;
                
            }

            double averageLoss = totalLoss / aggregatedProbabilities.Count;

            File.WriteAllText(fnLossSet, 
                JsonConvert.SerializeObject(lossSet, Formatting.Indented));
            File.WriteAllText(fnAverageLoss,averageLoss.ToString(CultureInfo.InvariantCulture));
        }

        /*
         * converts document level loss into term level loss, and applies min-max normalization. Applies
         * tf-idf scores (multiplies docLoss by the given term's tfIdf score) to produce the conversion.
         */
        private void DocumentLossToTermLoss(string fnLossSet, string fnTfIdf, string fnVocab, string fnTermLossSet)
        {
            Dictionary<string, Dictionary<string, double>> lossSet =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnLossSet);
            Dictionary<string, Dictionary<string, double>> tfIdf =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTfIdf);
            List<string> vocab = ReadFile.ReadJson<List<string>>(fnVocab);
            Dictionary<string, Dictionary<string, double>> termLossSet = 
                new Dictionary<string, Dictionary<string, double>>();

            foreach (var emotion in _emotions)
            {
                termLossSet[emotion] = new Dictionary<string, double>();

                foreach (var term in vocab)
                {
                    termLossSet[emotion][term] = 0.0;
                }
            }
            
            foreach (var emotion in lossSet.Keys)
            {
                var emotionDocuments = lossSet[emotion];

                for (int i = 0; i < emotionDocuments.Count; i++)
                {
                    var document = emotionDocuments.ElementAt(i);
                    double docLoss = document.Value;
                    Dictionary<string, double> docTfIdfs = tfIdf.ElementAt(i).Value;

                    foreach (string term in vocab)
                    {
                        double termLoss = 0.0;

                        if (docTfIdfs.TryGetValue(term, out var termTfIdf))
                        {
                            termLoss = docLoss * termTfIdf;
                        }

                        termLossSet[emotion][term] += termLoss;
                    }
                }
                
                double min = termLossSet[emotion].Min(t => t.Value); 
                double max = termLossSet[emotion].Max(t => t.Value);
                
                if (Math.Abs(min - max) > double.Epsilon)
                {
                    foreach (var term in termLossSet[emotion].Keys.ToList())
                    {
                        double normalizedTermLoss = (termLossSet[emotion][term] - min) / (max - min);
                        termLossSet[emotion][term] = normalizedTermLoss;
                    }
                }
            }
            
            File.WriteAllText(fnTermLossSet, 
                JsonConvert.SerializeObject(termLossSet, Formatting.Indented));
        }
        
        /*so, gradient descent time. I'm computing gradients of my parameters (weights and biases)
         *with respect to cost function, so that I can minimize the cost function
         * learning rate should be 0.001 iirc?
         */
        private void GradientDescent(string fnTermLossSet, double learningRate)
        {
            Dictionary<string, Dictionary<string, double>> termLossSet = 
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTermLossSet);

            foreach (var emotion in _emotions)
            {
                var termsCopy = new Dictionary<string, double>(_weights[emotion]);
                foreach (var term in termsCopy.Keys)
                {
                    double gradient = -termLossSet[emotion][term]; 
                    //this is meant to be a derivative of the loss function with respect to weight?
                    //but, the other things we might have applied, like our feature tfidf, were already applied
                    //before, in calculating the loss itself (or, at least in computing term loss out of doc loss)

                    _weights[emotion][term] -= learningRate * gradient;
                }
            }
        }
    }
}