using System;

namespace AIDA
{
    public static class Launch
    {

        /*
         * Main function, used to launch the various components of this pipeline.
         *
         * Notes:
         *  -A frequency threshold of 2 leaves a vocab of 39,587 words.
         */
        public static void Main(string[] args)
        {
            string fnStopWords = "../../stopwords-en.txt";
            int frequencyThreshold = 2;
            string fnChunks = "../../Chunks";
            int chunkSize = 10000;
            string fnTrainingData = null;
            string fnVocab = null;
            string fnCorpus = null;
            string fnTf = null;
            string fnIdf = null;
            string fnTfIdf = null;
            string fnProbabilities = null;
            string fnMlr = null;
            string fnMergedProbabilities = null;
            string fnAggregatedProbabilities = null;
            string fnLossSet = null;
            string fnAverageLoss = null;
            string fnTermLossSet = null;
            string fnInputGraphData = null;
            string fnOutputGraphData = null;
            double learningRate = 0.01;
            bool chooseFormat = true;

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
                        fnTrainingData = "../../JSONs/TrainingData.json";
                        fnVocab = "../../JSONs/Vocabulary.json";
                        fnCorpus = "../../JSONs/Corpus.json";
                        fnTf = "../../JSONs/TermFrequency.json";
                        fnIdf = "../../JSONs/InverseDocumentFrequency.json";
                        fnTfIdf = "../../JSONs/TF-IDF.json";
                        fnProbabilities = "../../JSONs/Probabilities.json";
                        fnMlr = "../../Saved MLRs/MLR.json";
                        fnMergedProbabilities = "../../JSONs/MergedProbabilities.json";
                        fnAggregatedProbabilities = "../../JSONs/AggregatedProbabilities.json";
                        fnLossSet = "../../JSONs/LossSet.json";
                        fnAverageLoss = "../../JSONs/AverageLoss.txt";
                        fnTermLossSet = "../../JSONs/TermLossSet.json";
                        fnInputGraphData = "../../Saved MLRs/MLR";
                        fnOutputGraphData = "../../Saved MLRs/graphData.csv";
                        chooseFormat = false;
                        break;
                    case "2":
                        Console.WriteLine("Partial chosen");
                        fnTrainingData = "../../Chunks/chunk_0.json";
                        fnVocab = "../../JSONs - Chunks/VocabularyChunk_0.json";
                        fnCorpus = "../../JSONs - Chunks/CorpusChunk_0.json";
                        fnTf = "../../JSONs - Chunks/TermFrequencyChunk_0.json";
                        fnIdf = "../../JSONs - Chunks/InverseDocumentFrequencyChunk_0.json";
                        fnTfIdf = "../../JSONs - Chunks/TF-IDFChunk_0.json";
                        fnProbabilities = "../../JSONs - Chunks/ProbabilitiesChunk_0.json";
                        fnMlr = "../../Saved MLRs - Chunks/MLRChunk_0.json";
                        fnMergedProbabilities = "../../JSONs - Chunks/MergedProbabilitiesChunk_0.json";
                        fnAggregatedProbabilities = "../../JSONs - Chunks/AggregatedProbabilitiesChunk_0.json";
                        fnLossSet = "../../JSONs - Chunks/LossSetChunk_0.json";
                        fnAverageLoss = "../../JSONs - Chunks/AverageLossChunk_0.txt";
                        fnTermLossSet = "../../JSONs - Chunks/TermLossSet.json";
                        fnInputGraphData = "../../Saved MLRs - Chunks/MLRChunk_0";
                        fnOutputGraphData = "../../Saved MLRs - Chunks/graphDataChunks.csv";
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
                Console.WriteLine("11. MLR - document aggregated probabilities");
                Console.WriteLine("12. MLR - loss set and average loss");
                Console.WriteLine("13. MLR - Document loss to term loss");
                Console.WriteLine("14. MLR - Gradient descent");
                Console.WriteLine("15. full MLR training process (ensure you have initialized first)");
                Console.WriteLine("16. Save MLR weight change vs iteration number graph data");
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
                        _ = new MultinomialLogisticRegression(fnVocab, fnMlr);
                        break;
                    case "9":
                        Console.WriteLine("Launching MLR - forward propagation and softmax");
                        choice = 0;
                        _ = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData, fnAggregatedProbabilities, 
                            fnLossSet, fnAverageLoss, fnVocab, fnTermLossSet, learningRate);
                        break;
                    case "10":
                        Console.WriteLine("Launching MLR - merge training documents and term probabilities");
                        choice = 1;
                        _ = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData, fnAggregatedProbabilities, 
                            fnLossSet, fnAverageLoss, fnVocab, fnTermLossSet, learningRate);
                        break;
                    case "11":
                        Console.WriteLine("Launching MLR - document aggregated probabilities");
                        choice = 2;
                        _ = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData, fnAggregatedProbabilities, 
                            fnLossSet, fnAverageLoss, fnVocab, fnTermLossSet, learningRate);
                        break;
                    case "12":
                        Console.WriteLine("Launching MLR - loss set and average loss");
                        choice = 3;
                        _ = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData, fnAggregatedProbabilities, 
                            fnLossSet, fnAverageLoss, fnVocab, fnTermLossSet, learningRate);
                        break;
                    case "13":
                        Console.WriteLine("Launching MLR - document loss to term loss");
                        choice = 4;
                        _ = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData, fnAggregatedProbabilities, 
                            fnLossSet, fnAverageLoss, fnVocab, fnTermLossSet, learningRate);
                        break;
                    case "14":
                        Console.WriteLine("Launching MLR - Gradient descent");
                        choice = 5;
                        _ = new MultinomialLogisticRegression(choice, fnMlr, fnTfIdf, fnProbabilities,
                            fnMergedProbabilities, fnCorpus, fnTrainingData, fnAggregatedProbabilities, 
                            fnLossSet, fnAverageLoss, fnVocab, fnTermLossSet, learningRate);
                        break;
                    case "15":
                        Console.WriteLine("Enter the number of iterations: ");
                        if (int.TryParse(Console.ReadLine(), out int iterations) && iterations > 0)
                        {
                            for (int i = 1; i <= iterations; i++)
                            {
                                Console.WriteLine($"Launching full MLR training process iteration {i} of {iterations}");
                                _ = new MultinomialLogisticRegression(fnMlr, fnTfIdf, fnProbabilities,
                                    fnMergedProbabilities, fnCorpus, fnTrainingData, fnAggregatedProbabilities, 
                                    fnLossSet, fnAverageLoss, fnVocab, fnTermLossSet, learningRate);
                                Console.WriteLine($"Full MLR training process iteration {i} of {iterations} finished");
                            }                            
                        }
                        else
                        {
                            Console.WriteLine("Invalid input. Please enter a valid positive integer.");
                        }
                        break;
                    case "16":
                        Console.WriteLine("Saving File");
                        SaveGraphData.SaveNetChangesToFile(fnInputGraphData, fnOutputGraphData);
                        break;
                    default:
                        Console.WriteLine("Invalid Option.");
                        break;
                }  
            }
        }
    }
}