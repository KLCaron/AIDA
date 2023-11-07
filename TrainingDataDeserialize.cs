using System;
using System.IO;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace AIDA
{
    public static class TrainingDataDeserialize
    {
        //reads JSON file, chunks it into smaller pieces
        public static void Chunkinator()
        {
            //absolute file location
            string inputFilePath = "../../Documents/training_data.json";
            string outputDirectory = "../../Chunks";
            int chunkSize = 10000; //how many lines should my new chunk be
            List<TrainingData> currentChunk = new List<TrainingData>();
            int chunkIndex = 0; //tracks where we're at in building our current chunk
            using (StreamReader fileReader = File.OpenText(inputFilePath))
            {
                string line;
                while ((line = fileReader.ReadLine()) != null) //so long as there are lines to read
                {
                    try
                    { 
                        //creates a 'record' of the info at a given line, adds it to our current chunk
                        TrainingData record = JsonConvert.DeserializeObject<TrainingData>(line);
                        currentChunk.Add(record);
                        //if we fill up this chunk, process it, empty it, and make a new one
                        if (currentChunk.Count >= chunkSize)
                        {
                            ProcessChunk(currentChunk, chunkIndex, outputDirectory);
                            currentChunk.Clear();
                            chunkIndex++;
                        }
                    }
                    //if there's an error, print it out
                    catch (JsonException ex)
                    {
                        Console.WriteLine($"Error parsing JSON: {ex.Message}");
                    }
                }
                //if our chunk isn't full, but we don't have any more lines, process it
                if (currentChunk.Count > 0)
                {
                    ProcessChunk(currentChunk, chunkIndex, outputDirectory);
                }
            }
        }
        //process a chunk; turn it into it's own JSON file
        private static void ProcessChunk(List<TrainingData> chunk, int chunkIndex, string outputDirectory)
        {
            string chunkIndexString = chunkIndex.ToString(); //number the file we're making
            string outputFileName = Path.Combine(outputDirectory, $"chunk_{chunkIndexString}.json");
            //serialize our chunk into a JSON file
            File.WriteAllText(outputFileName, JsonConvert.SerializeObject(chunk, Formatting.Indented));
            //let me know it worked
            Console.WriteLine($"Processed and saved chunk {chunkIndexString} to {outputFileName}");
        }
    }
}