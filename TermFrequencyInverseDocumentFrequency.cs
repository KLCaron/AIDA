using Newtonsoft.Json;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace AIDA
{
    public static class TermFrequencyInverseDocumentFrequency
    {
        //builds my corpus, token list, and vocabulary
        public static void BuildVocabTokensCorpus(string fnChunks, string namingConvention, 
            string fnStopWords, string fnVocab, string fnTokens, string fnCorpus)
        {
            string[] jsonChunks = Directory.GetFiles(fnChunks, namingConvention);
            List<string> stopWords = ReadFile.ReadTxt<List<string>>(fnStopWords);
            List<List<string>> fullVocab = new List<List<string>>();
            List<List<string>> tokenLists = new List<List<string>>();
            List<List<string>> corpus = new List<List<string>>();
            
            foreach (string jsonChunk in jsonChunks)
            {
                List<TrainingData> documents = ReadTrainingData(jsonChunk);
                List<string> documentStrings = documents.Select(data => data.Text).ToList();
                //gives me my list of individual words
                List<List<string>> tokenizedDocuments = Tokenize(documentStrings, stopWords);
                corpus.AddRange(tokenizedDocuments);
                //need a list of terms sans the stop words too
                List<string> tokenList = TokenLists(tokenizedDocuments);
                tokenLists.Add(tokenList);
                //gives me my list of unique vocabulary
                List<string> vocab = Vocabulary(tokenizedDocuments);
                fullVocab.Add(vocab);
            }
            
            File.WriteAllText(fnCorpus, JsonConvert.SerializeObject(corpus, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnCorpus}");
            List<string> finalTokenList = TokenLists(tokenLists);
            File.WriteAllText(fnTokens, JsonConvert.SerializeObject(finalTokenList, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnTokens}");
            List<string> finalVocab = Vocabulary(fullVocab);
            File.WriteAllText(fnVocab, JsonConvert.SerializeObject(finalVocab, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnVocab}");
        }

        //gets my term frequency and tosses it into a json
        public static void TermFrequency(string fnCorpus, string fnTf)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);

            using (StreamWriter file = File.CreateText(fnTf))
            using (JsonTextWriter writer = new JsonTextWriter(file))
            {
                JsonSerializer serializer = new JsonSerializer()
                {
                    Formatting = Formatting.Indented
                };
                writer.WriteStartObject();

                for (int i = 0; i < corpus.Count; i++)
                {
                    string documentKey = "Document" + (i + 1).ToString();
                    Dictionary<string, double> tf = CalculateTfForDocument(corpus[i]);
                    writer.WritePropertyName(documentKey);
                    serializer.Serialize(writer, tf);
                }
                
                writer.WriteEndObject();
            }
            
            Console.WriteLine($"Processed and saved {fnTf}");
        }

        //calculates tf for my tf function
        private static Dictionary<string, double> CalculateTfForDocument(List<string> tweetTokens)
        {
            Dictionary<string, double> tf = new Dictionary<string, double>();
            int totalTerms = tweetTokens.Count;

            foreach (string term in tweetTokens)
            {
                int termCount = tweetTokens.Count(t => t == term);
                tf[term] = (double)termCount / totalTerms;
            }

            return tf;
        }

        //calculates idf and stores it inside a json
        public static void InverseDocumentFrequency(string fnCorpus,
            string fnVocab, string fnIdf)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);
            List<string> vocabulary = ReadFile.ReadJson<List<string>>(fnVocab);
            int totalDocuments = corpus.Count;
            
            using (StreamWriter file = File.CreateText(fnIdf))
            using (JsonTextWriter writer = new JsonTextWriter(file))
            {
                writer.Formatting = Formatting.Indented;
                writer.WriteStartObject();

                foreach (string term in vocabulary)
                {
                    int documentsWithTerm = corpus.Count(doc => doc.Contains(term));
                    double idf = Math.Log((double)totalDocuments / (1 + documentsWithTerm));
                    
                    writer.WritePropertyName(term);
                    writer.WriteValue(idf);
                }
                
                writer.WriteEndObject();
            }

            Console.WriteLine($"Processed and saved {fnIdf}");
        }
        
        //this handles actually reading the training data doc
        private static List<TrainingData> ReadTrainingData(string jsonFilePath)
        {
            try
            {
                string jsonText = File.ReadAllText(jsonFilePath);
                List<TrainingData> documents = JsonConvert.DeserializeObject<List<TrainingData>>(jsonText);
                return documents;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error Reading {jsonFilePath}: {ex.Message}");
                return new List<TrainingData>();
            }
        }

        //need to turn the the lines into collections of individual words (tokens)
        //also need to set it to ditch all the stop words
        //right now, this just returns a list of every word in the document
        private static List<List<string>> Tokenize(List<string> documentStrings, List<string> stopWords)
        {
            List<List<string>> tokenizedDocuments = new List<List<string>>();
            foreach (string document in documentStrings)
            {
                string[] words = document.Split(' ');
                List<string> tokens = new List<string>();
                foreach (string word in words)
                {
                    if (!stopWords.Contains(word))
                    {
                        tokens.Add(word);
                    }
                }
                tokenizedDocuments.Add(tokens);
            }
            return tokenizedDocuments;
        }

        //need to get a vocab going; go through and record unique words
        //what I want to do is create a single large list of vocab items; meaning I want to make
        //a list of lists of strings, each one a vocab list, then squash em down to only uniques
        private static List<string> Vocabulary(List<List<string>> tokenizedDocuments)
        {
            List<string> vocabulary = tokenizedDocuments.SelectMany(tokens => tokens).Distinct().ToList();
            return vocabulary;
        }

        private static List<string> TokenLists(List<List<string>> tokenizedDocuments)
        {
            List<string> tokenList = tokenizedDocuments.SelectMany(tokens => tokens).ToList();
            return tokenList;
        }

        //calculates my tfidf scores and tosses them into a json
        public static void CalculateTfIdf(string fnTf, string fnIdf, string fnTfIdf)
        {
            Dictionary<string, Dictionary<string, double>> tfScores =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTf);
            Dictionary<string, double> idfValues = ReadFile.ReadJson<Dictionary<string, double>>(fnIdf);

            Dictionary<string, Dictionary<string, double>> tfIdfScores = new Dictionary<string, Dictionary<string, double>>();
            foreach (var document in tfScores)
            {
                string documentKey = document.Key;
                Dictionary<string, double> documentTfIdf = new Dictionary<string, double>();

                foreach (var termTf in document.Value)
                {
                    string term = termTf.Key;
                    double tf = termTf.Value;
                    double idf = idfValues.TryGetValue(term, out var value) ? value : 0.0;
                    double tfIdf = tf * idf;

                    documentTfIdf[term] = tfIdf;
                }

                tfIdfScores[documentKey] = documentTfIdf;
            }
            
            File.WriteAllText(fnTfIdf, JsonConvert.SerializeObject(tfIdfScores, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnTfIdf}");
        }

        public static void MergeTfIdfTraining(string fnTfIdf, string fnTrainingData, string fnTfIdfMerged)
        {
            Dictionary<string, Dictionary<string, double>> tfIdf =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTfIdf);
            List<Dictionary<string, object>> training = ReadFile.ReadJson<List<Dictionary<string, object>>>(fnTrainingData);
            List<Dictionary<string, object>> mergedData = new List<Dictionary<string, object>>();

            if (tfIdf.Count == training.Count)
            {
                for (int i = 0; i < training.Count; i++)
                {
                    Dictionary<string, object> document = training[i];
                    string documentKey = "Document" + (i + 1).ToString();

                    if (tfIdf.TryGetValue(documentKey, out var tfIdfScores))
                    {
                        document["tfidf_scores"] = tfIdfScores;
                        mergedData.Add(document);
                    }
                    else
                    {
                        Console.Write($"No TF-IDF data found for document key: {documentKey}");
                    }
                }
            }
            else
            {
                Console.WriteLine("Mismatch in number of documents");
            }
            File.WriteAllText(fnTfIdfMerged, JsonConvert.SerializeObject(mergedData, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnTfIdfMerged}");
        }
    }
}