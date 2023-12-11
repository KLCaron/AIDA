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

            try
            {
                List<Dictionary<string, string>> documents =
                    JsonConvert.DeserializeObject<List<Dictionary<string, string>>>(File.ReadAllText(fnTrainingData));

                foreach (var document in documents)
                {
                    currentChunk.Add(document);
                    if (currentChunk.Count >= chunkSize)
                    {
                        ProcessChunk(currentChunk, chunkIndex, fnChunks);
                        currentChunk.Clear();
                        chunkIndex++;
                    }
                }

                if (currentChunk.Count > 0)
                {
                    ProcessChunk(currentChunk, chunkIndex, fnChunks);
                }
            }
            catch (JsonException ex)
            {
                Console.WriteLine($"Error parsing {fnTrainingData}: {ex.Message}");
            }
        }

        private static void ProcessChunk(List<Dictionary<string, string>> chunk, int chunkIndex, string fnChunks)
        {
            string chunkIndexString = chunkIndex.ToString();
            string fnOutput = Path.Combine(fnChunks, $"chunk_{chunkIndexString}.json");
            
            File.WriteAllText(fnOutput, JsonConvert.SerializeObject(chunk, Formatting.Indented));
            Console.WriteLine($"Processed and saved chunk {chunkIndexString} to {fnChunks}");
        }
        
        //builds my corpus
        public static void Corpus(string fnTrainingData, string fnStopWords, string fnCorpus)
        {
            List<string> stopWords = ReadFile.ReadTxt(fnStopWords);
            
            List<List<string>> corpus = Tokenize(fnTrainingData, stopWords);
            
            File.WriteAllText(fnCorpus, JsonConvert.SerializeObject(corpus, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnCorpus}");
        }
        
        //used for corpus
        private static List<List<string>> Tokenize(string fnTrainingData, 
            List<string> stopWords)
        {
            List<Dictionary<string, string>> documentStrings =
                ReadFile.ReadJson<List<Dictionary<string, string>>>(fnTrainingData);
            List<List<string>> tokenizedDocuments = new List<List<string>>();
            List<int> indicesToRemove = new List<int>();

            for (int i = 0; i < documentStrings.Count; i++)
            {
                if (documentStrings[i].TryGetValue("text", out var text))
                {
                    string[] words = text.Split(' ');
                    List<string> tokens = new List<string>();

                    foreach (string word in words)
                    {
                        if (!stopWords.Contains(word) && !string.IsNullOrWhiteSpace(word))
                        {
                            tokens.Add(word);
                        }
                    }

                    if (tokens.Count > 0)
                    {
                        tokenizedDocuments.Add(tokens);
                    }
                    else
                    {
                        indicesToRemove.Add(i);
                    }
                }
            }

            for (int i = indicesToRemove.Count - 1; i >= 0; i--)
            {
                documentStrings.RemoveAt(indicesToRemove[i]);
            }

            File.WriteAllText(fnTrainingData, JsonConvert.SerializeObject(documentStrings, Formatting.Indented));
            Console.WriteLine($"Updated {fnTrainingData} to remove empty documents");
            return tokenizedDocuments;
        }

        public static void Vocabulary(string fnCorpus, string fnVocab)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);
            List<string> vocab = corpus.SelectMany(tokens => tokens).Distinct().ToList();
            
            File.WriteAllText(fnVocab, JsonConvert.SerializeObject(vocab, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnVocab}");
        }

        public static void AppendStopwords(string fnCorpus, string fnStopWords, int frequencyThreshold)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);

            Dictionary<string, int> wordFrequencies = new Dictionary<string, int>();
            foreach (var tokens in corpus)
            {
                foreach (var word in tokens)
                {
                    if (!wordFrequencies.ContainsKey(word))
                    {
                        wordFrequencies[word] = 1;
                    }
                    else
                    {
                        wordFrequencies[word]++;
                    }
                }
            }

            List<string> wordsBelowThreshold = wordFrequencies.Where(pair => pair.Value < frequencyThreshold)
                .Select(pair => pair.Key).ToList();
            
            File.AppendAllLines(fnStopWords, wordsBelowThreshold);
            Console.WriteLine($"Updated {fnStopWords} with words appearing fewer than " +
                              $"{frequencyThreshold.ToString()} times.");
        }

        //gets my term frequency and tosses it into a json
        public static void TermFrequency(string fnCorpus, string fnTf, string fnTrainingData)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);
            List<Dictionary<string, string>> documents =
                ReadFile.ReadJson<List<Dictionary<string, string>>>(fnTrainingData);

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
                    string emotion = documents[i]["emotions"];
                    Dictionary<string, double> tf = CalculateTfForDocument(corpus[i]);
                    writer.WritePropertyName($"{emotion}_{i.ToString()}");
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
        public static void TfIdf(string fnTf, string fnIdf, string fnTfIdf)
        {
            Dictionary<string, Dictionary<string, double>> tfScores =
                ReadFile.ReadJson<Dictionary<string, Dictionary<string, double>>>(fnTf);
            Dictionary<string, double> idfValues = ReadFile.ReadJson<Dictionary<string, double>>(fnIdf);

            using (StreamWriter file = File.CreateText(fnTfIdf))
            using (JsonTextWriter writer = new JsonTextWriter(file))
            {
                JsonSerializer serializer = new JsonSerializer()
                {
                    Formatting = Formatting.Indented
                };
                writer.WriteStartObject();

                for (int i = 0; i < tfScores.Count; i++)
                {
                    Dictionary<string, double> documentTfIdf = new Dictionary<string, double>();
                    var tfScoreValues = tfScores.ElementAt(i);

                    foreach (var termTf in tfScoreValues.Value)
                    {
                        string term = termTf.Key;
                        double tf = termTf.Value;
                        double idf = idfValues.TryGetValue(term, out var value) ? value : 0.0;
                        double tfIdf = tf * idf;

                        documentTfIdf[term] = tfIdf;
                        
                    }
                    
                    string emotion = tfScoreValues.Key;
                    writer.WritePropertyName(emotion);
                    serializer.Serialize(writer, documentTfIdf);
                    
                }

                writer.WriteEndObject();
                Console.WriteLine($"Processed and saved {fnTfIdf}");
            }
        }
    }
}