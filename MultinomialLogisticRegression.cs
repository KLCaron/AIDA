using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace AIDA
{
    public class MultinomialLogisticRegression
    {
        private Dictionary<string, Dictionary<string, double>> _weights;
        private Dictionary<string, double> _biases;
        private readonly string[] _emotions = { "sadness", "joy", "love", "anger", "fear", "surprise" };
        private readonly Random _rand;
        
        /*
         * Initialize a starting model object
         */
        public MultinomialLogisticRegression(string fnVocab, string fnMlr)
        {
            _rand = new Random();
            List<string> vocab = ReadFile.ReadJson<List<string>>(fnVocab);
            InitializeParameters(vocab);
            
            SaveModel(fnMlr);
            Console.WriteLine("MLR - initialize new object finished");
        }

        /*
         * Save the model object to a json, to re-use later
         */
        private void SaveModel(string fnMlr)
        {
            if (File.Exists(fnMlr))
            {
                int fileNumber = 1;
                string baseFileName = Path.GetFileNameWithoutExtension(fnMlr);
                string fileExtension = Path.GetExtension(fnMlr);
                string directory = Path.GetDirectoryName(fnMlr);
                
                string newFilename = fnMlr;

                if (directory != null)
                {
                    while (File.Exists(newFilename))
                    { 
                        newFilename = Path.Combine(directory, $"{baseFileName}_{fileNumber}{fileExtension}");
                        fileNumber++;
                    }
                    File.Move(fnMlr, newFilename);                    
                }
            }
            
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
         * Allows for specific work to be done on a saved model object.
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
                {5, () => GradientDescent(fnTermLossSet, fnLossSet, learningRate)}
            };

            if (actions.TryGetValue(choice, out Action action))
            {
                action();
            }

            SaveModel(fnMlr);
        }

        /*
         * Runs an entire iteration in one go on a saved model object.
         */
        public MultinomialLogisticRegression(string fnMlr, string fnTfIdf, string fnProbabilities, 
            string fnMergedProbabilities, string fnCorpus, string fnDocuments, string fnAggregatedProbabilities, 
            string fnLossSet, string fnAverageLoss, string fnVocab, string fnTermLossSet, double learningRate)
        {
            string jsonData = File.ReadAllText(fnMlr);
            var modelData = JsonConvert.DeserializeObject<dynamic>(jsonData);

            _weights = modelData.Weights.ToObject<Dictionary<string, Dictionary<string, double>>>();
            _biases = modelData.Biases.ToObject<Dictionary<string, double>>();

            int? randSeed = modelData.RandSeed;
            _rand = randSeed.HasValue ? new Random(randSeed.Value) : new Random();

            MlrForwardPropSoftMax(fnProbabilities, fnTfIdf);
            MergeDocumentsTermProbabilities(fnMergedProbabilities, fnCorpus, fnDocuments,
                fnProbabilities);
            AggregateDocumentProbabilities(fnAggregatedProbabilities, fnMergedProbabilities);
            CalcCrossEntropyLoss(fnAggregatedProbabilities, fnLossSet, fnAverageLoss);
            DocumentLossToTermLoss(fnLossSet, fnTfIdf, fnVocab, fnTermLossSet);
            GradientDescent(fnTermLossSet, fnLossSet, learningRate);
            SaveModel(fnMlr);
        }

        /*
         * Launches the forward propagation and softmax sections of the MLR process.
         */
        private void MlrForwardPropSoftMax(string fnProbabilities, string fnTfIdf)
        {
            Dictionary<string, Dictionary<string, double>> tfIdf =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTfIdf);
            
            Dictionary<string, Dictionary<string, double>> classScores = ForwardPropagation(tfIdf);
            Dictionary<string, Dictionary<string, double>> probabilities = SoftMax(classScores);

            File.WriteAllText(fnProbabilities, 
                JsonConvert.SerializeObject(probabilities, Formatting.Indented));
            Console.WriteLine("MLR - forward propagation and softmax finished");
        }

        /*
         * Initializes weights and biases
         */
        private void InitializeParameters(List<string> vocab)
        {
            _weights = InitializeWeights(vocab);
            _biases = InitializeBiases();
        }

        /*
         * Initializes weights to some random near-0 value.
         */
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

        /*
         * Initializes biases to some random near-0 value.
         */
        private Dictionary<string, double> InitializeBiases()
        {
            Dictionary<string, double> initialBiases = new Dictionary<string, double>();

            foreach (var emotion in _emotions)
            {
                initialBiases[emotion] = _rand.NextDouble() * 0.01;
            }

            return initialBiases;
        }

        /*
         * Performs forward propagation to calculate class scores based on TF-IDF values.
         *
         * Parameters:
         *   -tfIdf: TF-IDF values for terms in documents.
         *
         * Returns:
         *   -Dictionary<string, Dictionary<string, double>>: Class scores for each emotion label.
         *
         * Implementation Details:
         *   -Iterates through each emotion label's weights and biases.
         *   -Calculates scores for terms in TF-IDF if weight exists for the emotion.
         *   -Updates or adds scores to the emotion's dictionary in classScores.
         */
        private Dictionary<string, Dictionary<string, double>> ForwardPropagation(Dictionary<string, 
            Dictionary<string, double>> tfIdf)
        {
            Dictionary<string, Dictionary<string, double>> classScores = new Dictionary<string, 
                Dictionary<string, double>>();

            foreach (var emotion in _weights)
            {
                string emotionLabel = emotion.Key;
                Dictionary<string, double> emotionScores = new Dictionary<string, double>();
                double bias = _biases[emotionLabel];
        
                foreach (var tfIdfEmotion in tfIdf.Where(kv => 
                             kv.Key.StartsWith(emotionLabel.Split('_')[0])))
                {
                    foreach (var tfIdfTerm in tfIdfEmotion.Value)
                    {
                        string term = tfIdfTerm.Key;
                
                        if (emotion.Value.TryGetValue(term, out double weight))
                        {
                            double tfIdfValue = tfIdfTerm.Value;
                            double score = weight * tfIdfValue;

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

                classScores[emotionLabel] = emotionScores;
            }

            return classScores;
        }

        /*
         * Computes SoftMax probabilities based on class scores.
         *
         * Parameters:
         *   -classScores: Dictionary of class scores for each emotion label.
         *
         * Returns:
         *   -Dictionary<string, Dictionary<string, double>>: SoftMax probabilities for each term.
         *
         * Implementation Details:
         *   -Iterates through classScores to compute SoftMax probabilities for each term.
         *   -Calculates maximum score for normalization.
         *   -Computes SoftMax probabilities and stores them in 'probabilities' dictionary.
         */
        private Dictionary<string, Dictionary<string, double>> SoftMax(Dictionary<string, 
            Dictionary<string, double>> classScores)
        {
            Dictionary<string, Dictionary<string, double>> probabilities = new Dictionary<string, 
                Dictionary<string, double>>();

            foreach (var term in classScores.SelectMany(emotion => 
                         emotion.Value.Keys).Distinct())
            {
                Dictionary<string, double> termProbabilities = new Dictionary<string, double>();
                double maxScore = double.MinValue;

                foreach (var emotion in classScores)
                {
                    if (emotion.Value.TryGetValue(term, out double value) && value > maxScore)
                    {
                        maxScore = value;
                    }
                }

                double expSum = 0.0;

                foreach (var emotion in classScores)
                {
                    if (emotion.Value.TryGetValue(term, out double value))
                    {
                        double expScore = Math.Exp(value - maxScore);
                        expSum += expScore;
                        termProbabilities[emotion.Key] = expScore;
                    }
                }

                foreach (var emotion in termProbabilities)
                {
                    probabilities.TryGetValue(term, out Dictionary<string, double> existingProbabilities);
                    probabilities[term] = existingProbabilities ?? new Dictionary<string, double>();

                    probabilities[term][emotion.Key] = emotion.Value / expSum;
                }
            }

            return probabilities;
        }
        
        /*
         * Merges document term probabilities with training documents.
         *
         * Parameters:
         *   -fnMergedProbabilities: File path to store merged document probabilities.
         *   -fnCorpus: File path to the corpus data.
         *   -fnDocuments: File path to the documents data.
         *   -fnAllTermProbabilities: File path to all term probabilities data.
         *
         * Implementation Details:
         *   -Reads corpus, documents, and all term probabilities from JSON files.
         *   -Merges term probabilities with training documents based on the corpus.
         *   -Stores merged document probabilities in the specified file path.
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
                    new Dictionary<string, Dictionary<string, double>>
                    {
                        [trueEmotion] = null
                    };

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
            Console.WriteLine("Merge training documents and term probabilities finished");
        }
        
        /*
         * Aggregates document probabilities into a single set of probabilities without normalization.
         *
         * Parameters:
         *   -fnAggregatedProbabilities: File path to store aggregated document probabilities.
         *   -fnMergedProbabilities: File path to merged document probabilities.
         *
         * Implementation Details:
         *   -Reads merged document probabilities from JSON file.
         *   -Aggregates probabilities for each document into a single set based on emotion labels.
         *   -Stores aggregated probabilities in the specified file path.
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
                Dictionary<string, double> emotionSet = new Dictionary<string, double>
                {
                    [trueEmotionKey] = 0.0
                };

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
            Console.WriteLine("Document aggregated probabilities finished");
        }
        
        /*
         * Calculates cross-entropy loss based on aggregated probabilities and emotion labels.
         *
         * Parameters:
         *   -fnAggregatedProbabilities: File path to aggregated document probabilities.
         *   -fnLossSet: File path to store individual document losses.
         *   -fnAverageLoss: File path to store the average loss value.
         *
         * Implementation Details:
         *   -Reads aggregated document probabilities from JSON file.
         *   -Computes cross-entropy loss for each document based on predicted and true probabilities.
         *   -Stores individual document losses and the average loss value in specified file paths.
         */
        private void CalcCrossEntropyLoss(string fnAggregatedProbabilities, string fnLossSet, string fnAverageLoss)
        {
            Dictionary<string, Dictionary<string, double>> aggregatedProbabilities =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnAggregatedProbabilities);
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
            Console.WriteLine("Loss set and average loss finished");
        }
        
        /*
         * Calculates term-wise losses from document-level losses and TF-IDF values.
         *
         * Parameters:
         *   -fnLossSet: File path to individual document losses.
         *   -fnTfIdf: File path to TF-IDF values for terms.
         *   -fnVocab: File path to vocabulary data.
         *   -fnTermLossSet: File path to store term-wise losses per emotion.
         *
         * Implementation Details:
         *   -Reads individual document losses, TF-IDF values, and vocabulary from JSON files.
         *   -Computes term-wise losses for each emotion based on document losses and TF-IDF values.
         *   -Stores term-wise losses per emotion in the specified file path.
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
            }
            
            File.WriteAllText(fnTermLossSet, 
                JsonConvert.SerializeObject(termLossSet, Formatting.Indented));
            Console.WriteLine("Document loss to term loss finished");
        }
        
        /*
         * Performs gradient descent to update weights and biases.
         *
         * Parameters:
         *   -fnTermLossSet: File path to term-wise losses per emotion.
         *   -fnLossSet: File path to individual document losses per emotion.
         *   -learningRate: The learning rate used in gradient descent.
         *
         * Implementation Details:
         *   -Reads term-wise losses and individual document losses from JSON files.
         *   -Updates weights and biases for each emotion using gradient descent formula.
         *   -Adjusts weights and biases based on calculated gradients and learning rate.
         */
        private void GradientDescent(string fnTermLossSet, string fnLossSet, double learningRate)
        {
            Dictionary<string, Dictionary<string, double>> termLossSet = 
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTermLossSet);
            Dictionary<string, Dictionary<string, double>> lossSet =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnLossSet);

            foreach (var emotion in _emotions)
            {
                double biasGradient = 0.0;
                var termsCopy = new Dictionary<string, double>(_weights[emotion]);

                foreach (var term in termsCopy.Keys)
                {
                    double termGradient = -termLossSet[emotion][term];

                    _weights[emotion][term] -= learningRate * termGradient;
                }

                foreach (var document in lossSet[emotion].Keys)
                {
                    double documentGradient = -lossSet[emotion][document];

                    biasGradient += documentGradient;
                }

                _biases[emotion] -= biasGradient * learningRate;
            }
            
            Console.WriteLine("Gradient descent finished");
        }
    }
}