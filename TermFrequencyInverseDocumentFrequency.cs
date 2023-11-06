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
            Console.WriteLine("Processed and saved Corpus");
            List<string> finalTokenList = TokenLists(tokenLists);
            File.WriteAllText(outputFileNameTokens, JsonConvert.SerializeObject(finalTokenList, Formatting.Indented));
            Console.WriteLine("Processed and saved TokenList");
            List<string> finalVocab = Vocabulary(fullVocab);
            File.WriteAllText(outputFileNameVocab, JsonConvert.SerializeObject(finalVocab, Formatting.Indented));
            Console.WriteLine("Processed and saved Vocabulary");
        }

        //gets my term frequency and tosses it into a json
        public static void TermFrequency()
        {
            List<string> vocabulary = ReadJsonList("../../Vocabulary.json");
            List<List<string>> corpus = ReadJsonListList("../../Corpus.json");
            string outputFilename = "../../TermFrequency.json";

            List<Dictionary<string, double>> tf = CalculateTf(corpus, vocabulary);

            File.WriteAllText(outputFilename, JsonConvert.SerializeObject(tf, Formatting.Indented));
            Console.WriteLine("Processed and saved Term Frequency List");
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
                Console.WriteLine($"Error reading vocabulary: {ex.Message}");
                return new List<string>();
            }
        }

        //calculates tf for my tf function
        private static List<Dictionary<string, double>> CalculateTf(List<List<string>> corpus, List<string> vocabulary)
        {
            List<Dictionary<string, double>> tfList = new List<Dictionary<string, double>>();

            foreach (List<string> tweetTokens in corpus)
            {
                Dictionary<string, double> tf = new Dictionary<string, double>();
                int totalTerms = tweetTokens.Count;

                foreach (string term in vocabulary)
                {
                    int termCount = tweetTokens.Count(t => t == term);
                    tf[term] = (double)termCount / totalTerms;
                }
                tfList.Add(tf);
            }

            return tfList;
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
            Dictionary<string, double> idfValues = new Dictionary<string, double>();

            foreach (string term in vocabulary)
            {
                int documentsWithTerm = corpus.Count(doc => doc.Contains(term));
                double idf = Math.Log((double)totalDocuments / (1 + documentsWithTerm));
                idfValues[term] = idf;
            }
            
            File.WriteAllText(idfOutputFileName, JsonConvert.SerializeObject(idfValues, Formatting.Indented));
            Console.WriteLine("Processed and saved Inverse Document Frequency");
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
                Console.WriteLine($"Error reading vocabulary: {ex.Message}");
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
                Console.WriteLine($"Error Reading JSON file: {ex.Message}");
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
                Console.WriteLine($"Error reading stop words: {ex.Message}");
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
        private static List<Dictionary<string, double>> ReadJsonListDictionary(string jsonFilePath)
        {
            try
            {
                string jsonText = File.ReadAllText(jsonFilePath);
                List<Dictionary<string, double>> tf = JsonConvert.DeserializeObject<List<Dictionary<string, double>>>(jsonText);
                return tf;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading Term Frequency: {ex.Message}");
                return new List<Dictionary<string, double>>();
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
                Console.WriteLine($"Error reading Inverse Document Frequency: {ex.Message}");
                return new Dictionary<string, double>();
            }
        }

        public static void CalculateTfIdf()
        {
            string inputFileNameTf = "../../TermFrequency.json";
            string inputFilenameIdf = "../../InverseDocumentFrequency.json";
            
            List<Dictionary<string, double>> tfScores = ReadJsonListDictionary(inputFileNameTf);
            Dictionary<string, double> idfValues = ReadJsonDictionary(inputFilenameIdf);

            Dictionary<int, Dictionary<string, double>> tfIdfScores = new Dictionary<int, Dictionary<string, double>>();
            for (int i = 0; i < tfScores.Count; i++)
            {
                Dictionary<string, double> documentTfIdf = new Dictionary<string, double>();

                foreach (var kvp in tfScores[i])
                {
                    string term = kvp.Key;
                    double tf = kvp.Value;
                    double idf = idfValues[term];
                    double tfIdf = tf * idf;

                    documentTfIdf[term] = tfIdf;
                }

                tfIdfScores[i] = documentTfIdf;
            }

            string outputFilename = "../../TermFrequencyInverseDocumentFrequency.json";
            File.WriteAllText(outputFilename, JsonConvert.SerializeObject(tfIdfScores, Formatting.Indented));
            Console.WriteLine("Processed and saved Term Frequency - Inverse Document Frequency scores");
        }
    }
}