using System;

namespace AIDA
{
    public static class Launch
    {

        //launches all of my various calculators
        public static void Main(string[] args)
        {
            string fnStopWords = "../../stopwords-en.txt";
            string fnTrainingData = "../../Documents/training_data.json";
            string fnChunks = "../../Chunks";
            string namingConvention = "chunk_*.json";
            string fnVocab = "../../Vocabulary.json";
            string fnTokens = "../../TokenList.json";
            string fnCorpus = "../../Corpus.json";
            string fnTf = "../../TermFrequency.json";
            string fnIdf = "../../InverseDocumentFrequency.json";
            string fnTfIdf = "../../TF-IDF.json";
            string fnTfIdfMerged = "../../TF-IDFMerged.json";
            
            while (true)
            {
                Console.WriteLine("Select an option:");
                Console.WriteLine("1. Chunkinator");
                Console.WriteLine("2. BuildVocabCorpus");
                Console.WriteLine("3. Term Frequency");
                Console.WriteLine("4. Inverse Document Frequency");
                Console.WriteLine("5. Term Frequency - Inverse Document Frequency");
                Console.WriteLine("6. merge TF-IDF with training data");
                Console.WriteLine("q to quit.");
                string input = Console.ReadLine();

                if (input?.ToLower() == "q")
                {
                    break;
                }
                switch (input)
                {
                    case "1":
                        Console.WriteLine("Launching Chunkinator.");
                        TrainingDataDeserialize.Chunkinator();
                        break;
                    case "2":
                        Console.WriteLine("Launching BuildVocabCorpus");
                        TermFrequencyInverseDocumentFrequency.BuildVocabTokensCorpus(fnChunks,
                            namingConvention, fnStopWords, fnVocab, fnTokens, fnCorpus);
                        break;
                    case "3":
                        Console.WriteLine("Launching Term Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.TermFrequency(fnCorpus,
                            fnTf);
                        break;
                    case "4":
                        Console.WriteLine("Launching Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.InverseDocumentFrequency(fnCorpus, 
                            fnVocab, fnIdf);
                        break;
                    case "5":
                        Console.WriteLine("Launching Term Frequency - Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.CalculateTfIdf(fnTf, fnIdf, fnTfIdf);
                        break;
                    case "6":
                        Console.WriteLine("Launching TF-IDF merger");
                        TermFrequencyInverseDocumentFrequency.MergeTfIdfTraining(fnTfIdf, fnTrainingData, 
                            fnTfIdfMerged);
                        break;
                    default:
                        Console.WriteLine("Invalid Option.");
                        break;
                }  
            }
        }
    }
}