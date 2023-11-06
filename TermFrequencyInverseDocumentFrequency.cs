using Newtonsoft.Json;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace AIDA
{
    public static class TermFrequencyInverseDocumentFrequency
    {
        //set this up to more easily choose docs to read, and I guess manage this?
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
                List<TrainingData> documents = ReadDocumentsFromJson(jsonChunk);
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
            
            File.WriteAllText(outputFileNameCorpus, JsonConvert.SerializeObject(corpus));
            List<string> finalTokenList = TokenLists(tokenLists);
            File.WriteAllText(outputFileNameTokens, JsonConvert.SerializeObject(finalTokenList));
            Console.WriteLine("Processed and saved TokenList");
            List<string> finalVocab = Vocabulary(fullVocab);
            File.WriteAllText(outputFileNameVocab, JsonConvert.SerializeObject(finalVocab));
            Console.WriteLine("Processed and saved Vocabulary");
        }

        public static void TermFrequency()
        {
            List<string> vocabulary = ReadJSON("../../Vocabulary.json");
            List<string> tokenList = ReadJSON("../../TokenList.json");
            string outputFilename = "../../TermFrequency.json";

            Dictionary<string, double> tf = CalculateTF(tokenList, vocabulary);

            File.WriteAllText(outputFilename, JsonConvert.SerializeObject(tf, Formatting.Indented));
            Console.WriteLine("Processed and saved Term Frequency List");
        }

        private static List<string> ReadJSON(string jsonFilePath)
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

        private static Dictionary<string, double> CalculateTF(List<string> documentTokens, List<string> vocabulary)
        {
            Dictionary<string, double> tf = new Dictionary<string, double>();
            int totalTerms = documentTokens.Count;

            foreach (string term in vocabulary)
            {
                int termCount = documentTokens.Count(t => t == term);
                tf[term] = (double)termCount / totalTerms;
            }

            return tf;
        }

        public static void InverseDocumentFrequency()
        {
            string corpusFilePath = "../../Corpus.json";
            string vocabularyFilePath = "../../Vocabulary.json";
            string idfOutputFileName = "../../InverseDocumentFrequency.json";
            List<List<string>> corpus = LoadCorpus(corpusFilePath);
            List<string> vocabulary = ReadJSON(vocabularyFilePath);
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
        
        private static List<List<string>> LoadCorpus(string jsonFilePath)
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
        
        //this handles actually reading the doc we pick
        private static List<TrainingData> ReadDocumentsFromJson(string jsonFilePath)
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
        
        private static Dictionary<string, double> LoadTFValues()
        {
            string jsonFilePath = "../../TermFrequency.json";
            
            try
            {
                Dictionary<string, double> tfValues = File.ReadAllText(jsonFilePath);
                List<List<string>> corpus = JsonConvert.DeserializeObject<List<List<string>>>(jsonText);
                return corpus;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading vocabulary: {ex.Message}");
                return new List<List<string>>();
            }
        }

        private static Dictionary<string, double> LoadIDFValues()
        {
            
        }
    }
}