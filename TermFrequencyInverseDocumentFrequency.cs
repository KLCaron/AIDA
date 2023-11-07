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
        public static void BuildVocabCorpus()
        {
            string directoryPath = "../../Chunks";
            string namingConvention = "chunk_*.json";
            string[] jsonChunks = Directory.GetFiles(directoryPath, namingConvention);
            List<string> stopWords = ReadStopWords("../../stopwords-en.txt");
            List<List<string>> fullVocab = new List<List<string>>();
            List<List<string>> tokenLists = new List<List<string>>();
            List<List<string>> corpus = new List<List<string>>();
            string outputFileNameVocab = "../../Vocabulary.json";
            string outputFileNameTokens = "../../TokenList.json";
            string outputFileNameCorpus = "../../Corpus.json";
            
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
            
            File.WriteAllText(outputFileNameCorpus, JsonConvert.SerializeObject(corpus, Formatting.Indented));
            Console.WriteLine($"Processed and saved {outputFileNameCorpus}");
            List<string> finalTokenList = TokenLists(tokenLists);
            File.WriteAllText(outputFileNameTokens, JsonConvert.SerializeObject(finalTokenList, Formatting.Indented));
            Console.WriteLine($"Processed and saved {outputFileNameTokens}");
            List<string> finalVocab = Vocabulary(fullVocab);
            File.WriteAllText(outputFileNameVocab, JsonConvert.SerializeObject(finalVocab, Formatting.Indented));
            Console.WriteLine($"Processed and saved {outputFileNameVocab}");
        }

        //gets my term frequency and tosses it into a json
        public static void TermFrequency()
        {
            List<List<string>> corpus = ReadJsonListList("../../Corpus.json");
            string outputFilename = "../../TermFrequency.json";

            using (StreamWriter file = File.CreateText(outputFilename))
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
            
            Console.WriteLine($"Processed and saved {outputFilename}");
        }

        //reads a list of strings out of a json
        private static List<string> ReadJsonList(string jsonFilePath)
        {
            try
            {
                string jsonText = File.ReadAllText(jsonFilePath);
                List<string> vocabulary = JsonConvert.DeserializeObject<List<string>>(jsonText);
                return vocabulary;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {jsonFilePath}: {ex.Message}");
                return new List<string>();
            }
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
        public static void InverseDocumentFrequency()
        {
            string corpusFilePath = "../../Corpus.json";
            string vocabularyFilePath = "../../Vocabulary.json";
            string idfOutputFileName = "../../InverseDocumentFrequency.json";
            List<List<string>> corpus = ReadJsonListList(corpusFilePath);
            List<string> vocabulary = ReadJsonList(vocabularyFilePath);
            int totalDocuments = corpus.Count;
            
            using (StreamWriter file = File.CreateText(idfOutputFileName))
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

            Console.WriteLine($"Processed and saved {idfOutputFileName}");
        }
        
        private static List<List<string>> ReadJsonListList(string jsonFilePath)
        {
            try
            {
                string jsonText = File.ReadAllText(jsonFilePath);
                List<List<string>> corpus = JsonConvert.DeserializeObject<List<List<string>>>(jsonText);
                return corpus;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {jsonFilePath}: {ex.Message}");
                return new List<List<string>>();
            }
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

        //read stopwords from my txt file, can update that as needed (yoinked my list from NLTK)
        //actually yoinked em from another master stopwords list
        private static List<string> ReadStopWords(string filePath)
        {
            List<string> stopWords = new List<string>();

            try
            {
                using (StreamReader reader = new StreamReader(filePath))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        stopWords.Add(line);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {filePath}: {ex.Message}");
            }

            return stopWords;
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

        //reads my term frequency document
        private static Dictionary<string, Dictionary<string, double>> ReadJsonDictionaryDictionary(string jsonFilePath)
        {
            try
            {
                string jsonText = File.ReadAllText(jsonFilePath);
                Dictionary<string, Dictionary<string, double>> tf = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, double>>>(jsonText);
                return tf;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {jsonFilePath}: {ex.Message}");
                return new Dictionary<string, Dictionary<string, double>>();
            }
        }

        //reads my Inverse Document Frequency Document
        private static Dictionary<string, double> ReadJsonDictionary(string jsonFilePath)
        {
            try
            {
                string jsonText = File.ReadAllText(jsonFilePath);
                Dictionary<string, double> idf = JsonConvert.DeserializeObject<Dictionary<string, double>>(jsonText);
                return idf;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {jsonFilePath}: {ex.Message}");
                return new Dictionary<string, double>();
            }
        }

        private static List<Dictionary<string, object>> ReadJsonListDictionary(string jsonFilePath)
        {
            try
            {
                List<Dictionary<string, object>> training = new List<Dictionary<string, object>>();

                using (StreamReader file = new StreamReader(jsonFilePath))
                {
                    string line;
                    while ((line = file.ReadLine()) != null)
                    {
                        var jsonObject = JsonConvert.DeserializeObject<Dictionary<string, object>>(line);
                        training.Add(jsonObject);
                    }
                }
                return training;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {jsonFilePath}: {ex.Message}");
                return new List<Dictionary<string, object>>();
            }
        }

        //calculates my tfidf scores and tosses them into a json
        public static void CalculateTfIdf()
        {
            string inputFileNameTf = "../../TermFrequency.json";
            string inputFileNameIdf = "../../InverseDocumentFrequency.json";
            string outputFileName = "../../TF-IDF.json";
            
            Dictionary<string, Dictionary<string, double>> tfScores = ReadJsonDictionaryDictionary(inputFileNameTf);
            Dictionary<string, double> idfValues = ReadJsonDictionary(inputFileNameIdf);

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
            
            File.WriteAllText(outputFileName, JsonConvert.SerializeObject(tfIdfScores, Formatting.Indented));
            Console.WriteLine($"Processed and saved {outputFileName}");
        }

        public static void MergeTfIdfTraining()
        {
            string inputFileNameTfIdf = "../../TF-IDF.json";
            string inputFileNameTraining = "../../Documents/training_data.json";
            string outputFileName = "../../TF-IDFMerged.json";
            Dictionary<string, Dictionary<string, double>> tfIdf = ReadJsonDictionaryDictionary(inputFileNameTfIdf);
            List<Dictionary<string, object>> training = ReadJsonListDictionary(inputFileNameTraining);
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
            File.WriteAllText(outputFileName, JsonConvert.SerializeObject(mergedData, Formatting.Indented));
            Console.WriteLine($"Processed and saved {outputFileName}");
        }
    }
}