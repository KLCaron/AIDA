using System;

namespace AIDA
{
    public static class Launch
    {

        //launches all of my various calculators
        public static void Main(string[] args)
        {
            while (true)
            {
                Console.WriteLine("Select an option:");
                Console.WriteLine("1. Chunkinator");
                Console.WriteLine("2. BuildVocabCorpus");
                Console.WriteLine("3. Term Frequency");
                Console.WriteLine("4. Inverse Document Frequency");
                Console.WriteLine("5. Term Frequency - Inverse Document Frequency");
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
                        TermFrequencyInverseDocumentFrequency.BuildVocabCorpus();
                        break;
                    case "3":
                        Console.WriteLine("Launching Term Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.TermFrequency();
                        break;
                    case "4":
                        Console.WriteLine("Launching Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.InverseDocumentFrequency();
                        break;
                    case "5":
                        Console.WriteLine("Launching Term Frequency - Inverse Document Frequency Calculator");
                        TermFrequencyInverseDocumentFrequency.CalculateTfIdf();
                        break;
                    default:
                        Console.WriteLine("Invalid Option.");
                        break;
                }  
            }
        }
    }
}