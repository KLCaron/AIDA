using Newtonsoft.Json;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace AIDA
{
    public static class TermFrequencyInverseDocumentFrequency
    {
        public static void JsonChunker(string fnTrainingData, string fnChunks, int chunkSize)
        {
            List<Dictionary<string, string>> currentChunk = new List<Dictionary<string, string>>();
            int chunkIndex = 0;

            using (StreamReader fileReader = File.OpenText(fnTrainingData))
            {
                string line;
                while ((line = fileReader.ReadLine()) != null)
                {
                    try
                    {
                        Dictionary<string, string> document = ReadFile.ReadJson<Dictionary<string, string>>(line);
                        currentChunk.Add(document);
                        if (currentChunk.Count >= chunkSize)
                        {
                            ProcessChunk(currentChunk, chunkIndex, fnChunks);
                            currentChunk.Clear();
                            chunkIndex++;
                        }
                    }
                    catch (JsonException ex)
                    {
                        Console.WriteLine($"Error parsing {fnTrainingData}: {ex.Message}");
                    }
                }

                if (currentChunk.Count > 0)
                {
                    ProcessChunk(currentChunk, chunkIndex, fnChunks);
                }
            }
        }

        private static void ProcessChunk(List<Dictionary<string, string>> chunk, int chunkIndex, string fnChunks)
        {
            string chunkIndexString = chunkIndex.ToString();
            string fnOutput = Path.Combine(fnChunks, $"chunk_{chunkIndexString}.json");
            
            File.WriteAllText(fnOutput, JsonConvert.SerializeObject(chunk, Formatting.Indented));
            Console.WriteLine($"Processed and saved chunk {chunkIndexString} to {fnChunks}");
        }
        
        //builds my corpus, token list, and vocabulary
        public static void Corpus(string fnTrainingData, string fnStopWords, string fnCorpus)
        {
            List<string> stopWords = ReadFile.ReadTxt<List<string>>(fnStopWords);
            List<Dictionary<string, string>> documents =
                ReadFile.ReadJson<List<Dictionary<string, string>>>(fnTrainingData);
            
            List<List<string>> corpus = Tokenize(documents, stopWords);
            
            File.WriteAllText(fnCorpus, JsonConvert.SerializeObject(corpus, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnCorpus}");
        }
        
        //need to turn the the lines into collections of individual words (tokens)
        //also need to set it to ditch all the stop words
        //right now, this just returns a list of every word in the document
        private static List<List<string>> Tokenize(List<Dictionary<string, string>> documentStrings, List<string> stopWords)
        {
            List<List<string>> tokenizedDocuments = new List<List<string>>();
            foreach (var document in documentStrings)
            {
                if (document.TryGetValue("text", out var text))
                {
                    string[] words = text.Split(' ');
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
            }
            return tokenizedDocuments;
        }

        public static void Vocabulary(string fnCorpus, string fnVocab)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);
            List<string> vocab = corpus.SelectMany(tokens => tokens).Distinct().ToList();
            
            File.WriteAllText(fnVocab, JsonConvert.SerializeObject(vocab, Formatting.Indented));
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