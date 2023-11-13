using System;

namespace AIDA
{
    public static class Launch
    {

        //launches all of my various calculators
        public static void Main(string[] args)
        {
            string fnStopWords = "../../stopwords-en.txt";
            //swap these around to test functions on smaller set
            //string fnTrainingData = "../../Chunks/chunk_0.json";
            string fnTrainingData = "../../Documents/training_data.json";
            string fnChunks = "../../Chunks";
            int chunkSize = 1000;
            //string namingConvention = "chunk_*.json";
            string fnVocab = "../../Vocabulary.json";
            string fnCorpus = "../../Corpus.json";
            string fnTf = "../../IgnoredFiles/TermFrequency.json";
            string fnIdf = "../../InverseDocumentFrequency.json";
            string fnTfIdf = "../../TF-IDF.json";
            string fnTfIdfMerged = "../../IgnoredFiles/TF-IDFMerged.json";
            
            while (true)
            {
                Console.WriteLine("Select an option:");
                Console.WriteLine("1. JSON Chunker");
                Console.WriteLine("2. Corpus");
                Console.WriteLine("3. Vocabulary");
                Console.WriteLine("4. Term Frequency");
                Console.WriteLine("5. Inverse Document Frequency");
                Console.WriteLine("6. Term Frequency - Inverse Document Frequency");
                Console.WriteLine("7. merge TF-IDF with training data");
                Console.WriteLine("q to quit.");
                string input = Console.ReadLine();

                if (input?.ToLower() == "q")
                {
                    break;
                }
                switch (input)
                {
                    case "1":
                        Console.WriteLine("Launching JSON Chunker");
                        TermFrequencyInverseDocumentFrequency.JsonChunker(fnTrainingData, fnChunks, chunkSize);
                        break;
                    case "2":
                        Console.WriteLine("Launching Corpus builder");
                        TermFrequencyInverseDocumentFrequency.Corpus(fnTrainingData, fnStopWords, fnCorpus);
                        break;
                    case "3":
                        Console.WriteLine("Launching Vocabulary builder");
                        TermFrequencyInverseDocumentFrequency.Vocabulary(fnCorpus, fnVocab);
                        break;
                    case "4":
                        Console.WriteLine("Launching Term Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.TermFrequency(fnCorpus,
                            fnTf);
                        break;
                    case "5":
                        Console.WriteLine("Launching Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.InverseDocumentFrequency(fnCorpus, fnVocab, fnIdf);
                        break;
                    case "6":
                        Console.WriteLine("Launching Term Frequency - Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.CalculateTfIdf(fnTf, fnIdf, fnTfIdf);
                        break;
                    case "7":
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