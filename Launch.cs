using System;

namespace AIDA
{
    public static class Launch
    {

        //launches all of my various calculators - do not need merged one anymore, was kinda stupid to ever have tbqh
        public static void Main(string[] args)
        {
            string fnStopWords = "../../stopwords-en.txt";
            //10 leaves me with a vocab of 14,351, 5 leaves me with 53,748 stopwords and vocab of 21,710
            //7 leaves me with 57,645 stopwords and a vocab of 16,444
            //8 59,013 stopwords
            //2 39,587 words in vocab
            int frequencyThreshold = 2;
            string fnChunks = "../../Chunks";
            int chunkSize = 1000;
            //string namingConvention = "chunk_*.json";
            //number of classes, so how many emotions
            string fnTrainingData = null;
            string fnVocab = null;
            string fnCorpus = null;
            string fnTf = null;
            string fnIdf = null;
            string fnTfIdf = null;
            string fnProbabilities = null;
            string fnMlr = null;
            string fnMergedProbabilities = null;
            bool chooseFormat = true;
            MultinomialLogisticRegression mlr;

            while (chooseFormat)
            {
                Console.WriteLine("Choose a format:");
                Console.WriteLine("1. Full");
                Console.WriteLine("2. Partial");
                string input = Console.ReadLine();
                switch (input)
                {
                    case "1":
                        Console.WriteLine("Full chosen");
                        fnTrainingData = "../../Documents/training_data.json";
                        fnVocab = "../../Vocabulary.json";
                        fnCorpus = "../../Corpus.json";
                        fnTf = "../../IgnoredFiles/TermFrequency.json";
                        fnIdf = "../../InverseDocumentFrequency.json";
                        fnTfIdf = "../../TF-IDF.json";
                        fnProbabilities = "../../Probabilities.json";
                        fnMlr = "../../MLR.json";
                        fnMergedProbabilities = "../../MergedProbabilities.json";
                        chooseFormat = false;
                        break;
                    case "2":
                        Console.WriteLine("Partial chosen");
                        fnTrainingData = "../../Chunks/chunk_0.json";
                        fnVocab = "../../VocabularyChunk_0.json";
                        fnCorpus = "../../CorpusChunk_0.json";
                        fnTf = "../../TermFrequencyChunk_0.json";
                        fnIdf = "../../InverseDocumentFrequencyChunk_0.json";
                        fnTfIdf = "../../TF-IDFChunk_0.json";
                        fnProbabilities = "../../ProbabilitiesChunk_0.json";
                        fnMlr = "../../MLRChunk_0.json";
                        fnMergedProbabilities = "../../MergedProbabilitiesChunk_0.json";
                        chooseFormat = false;
                        break;
                    default:
                        Console.WriteLine("Invalid Option.");
                        break;
                }
            }
            
            while (true)
            {
                Console.WriteLine("Select an option:");
                Console.WriteLine("1. JSON Chunker");
                Console.WriteLine("2. Corpus");
                Console.WriteLine("3. Append stopwords to include words appearing fewer than " +
                                  $"{frequencyThreshold.ToString()} times");
                Console.WriteLine("4. Vocabulary");
                Console.WriteLine("5. Term Frequency");
                Console.WriteLine("6. Inverse Document Frequency");
                Console.WriteLine("7. Term Frequency - Inverse Document Frequency");
                Console.WriteLine("8. MLR - initialize new object");
                Console.WriteLine("9. MLR - forward propagation and softmax");
                Console.WriteLine("10. MLR - merge training documents and term probabilities");
                Console.WriteLine("q to quit.");
                string input = Console.ReadLine();

                if (input?.ToLower() == "q")
                {
                    break;
                }

                int choice;
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
                        Console.WriteLine("Launching StopWords Append");
                        TermFrequencyInverseDocumentFrequency.AppendStopwords(fnCorpus, fnStopWords, 
                            frequencyThreshold);
                        break;
                    case "4":
                        Console.WriteLine("Launching Vocabulary builder");
                        TermFrequencyInverseDocumentFrequency.Vocabulary(fnCorpus, fnVocab);
                        break;
                    case "5":
                        Console.WriteLine("Launching Term Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.TermFrequency(fnCorpus,
                            fnTf, fnTrainingData);
                        break;
                    case "6":
                        Console.WriteLine("Launching Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.InverseDocumentFrequency(fnCorpus, fnVocab, fnIdf);
                        break;
                    case "7":
                        Console.WriteLine("Launching Term Frequency - Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.TfIdf(fnTf, fnIdf, fnTfIdf);
                        break;
                    case "8":
                        Console.WriteLine("Launching MLR - initialize new object");
                        mlr = new MultinomialLogisticRegression(fnVocab, fnMlr);
                        break;
                    case "9":
                        Console.WriteLine("Launching MLR - forward propagation and softmax");
                        choice = 0;
                        mlr = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData);
                        break;
                    case "10":
                        Console.WriteLine("Launching MLR - merge training documents and term probabilities");
                        choice = 1;
                        mlr = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData);
                        break;
                    default:
                        Console.WriteLine("Invalid Option.");
                        break;
                }  
            }
        }
    }
}