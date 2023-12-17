using Newtonsoft.Json;
using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;

namespace AIDA
{
    public static class TermFrequencyInverseDocumentFrequency
    {
        /*
         * Divides the training data into chunks and processes each chunk.
         *
         * Parameters:
         *   -fnTrainingData: File path to the training data JSON file.
         *   -fnChunks: Directory path to store the chunked data.
         *   -chunkSize: Size limit for each chunk.
         *
         * Implementation Details:
         *   -Reads the training data from the provided file path.
         *   -Divides the data into chunks based on the specified chunk size.
         *   -Processes each chunk and stores it in the designated directory path.
         *   -Handles JSON parsing exceptions that might occur during data processing.
         */
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

        /*
         * Processes and saves a chunk of data to a JSON file.
         *
         * Parameters:
         *   -chunk: Chunk of data to be processed and saved.
         *   -chunkIndex: Index of the current chunk being processed.
         *   -fnChunks: Directory path to store the processed chunks.
         *
         * Implementation Details:
         *   -Converts the provided chunk into a JSON string with an indented format.
         *   -Saves the JSON string representing the chunk to a file in the specified directory.
         *   -Displays a console message indicating the successful processing and saving of the chunk.
         */
        private static void ProcessChunk(List<Dictionary<string, string>> chunk, int chunkIndex, string fnChunks)
        {
            string chunkIndexString = chunkIndex.ToString();
            string fnOutput = Path.Combine(fnChunks, $"chunk_{chunkIndexString}.json");
            
            File.WriteAllText(fnOutput, JsonConvert.SerializeObject(chunk, Formatting.Indented));
            Console.WriteLine($"Processed and saved chunk {chunkIndexString} to {fnChunks}");
        }
        
        /*
         * Generates a corpus from the training data by tokenizing and filtering stop words.
         *
         * Parameters:
         *   -fnTrainingData: File path to the training data.
         *   -fnStopWords: File path to the stop words list.
         *   -fnCorpus: File path to save the generated corpus.
         *
         * Implementation Details:
         *   -Reads stop words from the specified file path.
         *   -Tokenizes the training data, filtering out stop words.
         *   -Converts the resulting corpus into a JSON string with an indented format.
         *   -Saves the JSON string representing the corpus to the provided file path.
         *   -Displays a console message indicating the successful generation and saving of the corpus.
         */
        public static void Corpus(string fnTrainingData, string fnStopWords, string fnCorpus)
        {
            List<string> stopWords = ReadFile.ReadTxt(fnStopWords);
            
            List<List<string>> corpus = Tokenize(fnTrainingData, stopWords);
            
            File.WriteAllText(fnCorpus, JsonConvert.SerializeObject(corpus, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnCorpus}");
        }
        
        /*
         * Tokenizes the documents, removing stop words and empty documents, updating the data file accordingly.
         *
         * Parameters:
         *   -fnTrainingData: File path to the training data.
         *   -stopWords: List of words to be filtered out during tokenization.
         *
         * Returns:
         *   -List of tokenized documents after stop word removal.
         *
         * Implementation Details:
         *   -Reads and processes the document strings from the provided file path.
         *   -Iterates through each document to tokenize the text by splitting into words.
         *   -Filters out stop words and empty words from the tokens.
         *   -Updates the data file by removing empty documents.
         *   -Displays a console message indicating the successful update of the data file.
         *   -Returns the list of tokenized documents after stop word removal.
         */
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

        /*
         * Generates the vocabulary from the given corpus and saves it to a file.
         *
         * Parameters:
         *   -fnCorpus: File path to the corpus data.
         *   -fnVocab: File path to save the generated vocabulary.
         *
         * Implementation Details:
         *   -Reads the corpus data from the provided file path.
         *   -Flattens the list of tokenized documents into a single collection of tokens.
         *   -Removes duplicates to create a distinct vocabulary set.
         *   -Saves the generated vocabulary to the specified file path in JSON format.
         *   -Displays a console message confirming the successful processing and saving of the vocabulary.
         */
        public static void Vocabulary(string fnCorpus, string fnVocab)
        {
            List<List<string>> corpus = ReadFile.ReadJson<List<List<string>>>(fnCorpus);
            List<string> vocab = corpus.SelectMany(tokens => tokens).Distinct().ToList();
            
            File.WriteAllText(fnVocab, JsonConvert.SerializeObject(vocab, Formatting.Indented));
            Console.WriteLine($"Processed and saved {fnVocab}");
        }

        /*
         * Generates the vocabulary from the given corpus and saves it to a file.
         *
         * Parameters:
         *   -fnCorpus: File path to the corpus data.
         *   -fnVocab: File path to save the generated vocabulary.
         *
         * Implementation Details:
         *   -Reads the corpus data from the provided file path.
         *   -Flattens the list of tokenized documents into a single collection of tokens.
         *   -Removes duplicates to create a distinct vocabulary set.
         *   -Saves the generated vocabulary to the specified file path in JSON format.
         *   -Displays a console message confirming the successful processing and saving of the vocabulary.
         */
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
        
        /*
         * Computes term frequency (TF) for documents in the corpus and saves the TF values to a file.
         *
         * Parameters:
         *   -fnCorpus: File path to the corpus data.
         *   -fnTf: File path to save the term frequency data.
         *   -fnTrainingData: File path to the training data containing document emotions.
         *
         * Implementation Details:
         *   -Reads the corpus data and the training data from the provided file paths.
         *   -Calculates term frequency (TF) for each document in the corpus.
         *   -Writes the computed TF values to a JSON file, organized by document and emotion.
         *   -Displays a console message confirming the processing and saving of the TF data.
         */
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
        
        /*
         * Calculates the term frequency (TF) for a given document.
         *
         * Parameters:
         *   -tweetTokens: List of tokens representing a document.
         *
         * Returns:
         *   -Dictionary<string, double>: The computed term frequency for each term in the document.
         *
         * Implementation Details:
         *   -Initializes an empty dictionary to store term frequencies.
         *   -Counts the occurrence of each term in the document.
         *   -Calculates the term frequency by dividing the term count by the total number of terms in the document.
         *   -Returns the computed term frequency dictionary.
         */
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
        
        /*
         * Calculates the inverse document frequency (IDF) for terms in the given corpus.
         *
         * Parameters:
         *   -fnCorpus: Filepath to the corpus data.
         *   -fnVocab: Filepath to the vocabulary data.
         *   -fnIdf: Filepath to save the IDF values.
         *
         * Implementation Details:
         *   -Reads the corpus and vocabulary data from JSON files.
         *   -Computes IDF for each term in the vocabulary.
         *   -Writes the calculated IDF values to the specified output file.
         */
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
        
        /*
         * Computes the TF-IDF scores for terms in the given documents.
         *
         * Parameters:
         *   -fnTf: Filepath to the TF scores data.
         *   -fnIdf: Filepath to the IDF values.
         *   -fnTfIdf: Filepath to save the computed TF-IDF scores.
         *
         * Implementation Details:
         *   -Reads TF scores and IDF values from JSON files.
         *   -Calculates TF-IDF scores for each term in the TF scores.
         *   -Writes the computed TF-IDF scores to the specified output file.
         */
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