using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace AIDA
{
    public static class SaveGraphData
    {
        /*
         * Computes and saves net changes in weights to an output file.
         *
         * Parameters:
         *   -basePath: Base path for the model data files
         *   -fnOutput: File path to save the net changes
         *
         * Implementation Details:
         *   -Iterates through model data files in the specified base path
         *   -Computes net changes in weights between consecutive model data
         *   -Saves the computed net changes to the specified output file path
         */
        public static void SaveNetChangesToFile(string basePath, string fnOutput)
        {
            List<double> netChanges = new List<double>();

            Dictionary<string, Dictionary<string, double>> previousWeights = null;

            int i = 1;
            string currentFile = basePath + $"_{i}.json";
    
            while (File.Exists(currentFile))
            {
                string jsonData = File.ReadAllText(currentFile);
                var modelData = JsonConvert.DeserializeObject<dynamic>(jsonData);
                var weights = modelData.Weights.ToObject<Dictionary<string, Dictionary<string, double>>>();

                if (previousWeights != null)
                {
                    double netChange = CalculateNetChange(previousWeights, weights);
                    netChanges.Add(netChange);
                }

                previousWeights = weights;
                i++;
                currentFile = basePath + $"_{i}.json";
            }

            string lastFile = basePath + ".json";
            string lastJsonData = File.ReadAllText(lastFile);
            var lastModelData = JsonConvert.DeserializeObject<dynamic>(lastJsonData);
            var lastWeights = lastModelData.Weights.ToObject<Dictionary<string, Dictionary<string, double>>>();

            if (previousWeights != null)
            {
                double netChange = CalculateNetChange(previousWeights, lastWeights);
                netChanges.Add(netChange);
            }

            SaveDataToFile(netChanges, fnOutput);
        }

        /*
         * Calculates the net change in weights between two sets of weight values.
         *
         * Parameters:
         *   -previousWeights: Dictionary of previous weights for each emotion and term.
         *   -currentWeights: Dictionary of current weights for each emotion and term.
         *
         * Returns:
         *   -double: Net change in weights computed between the two sets of weights.
         *
         * Implementation Details:
         *   -Compares corresponding weights for each emotion and term in the two sets.
         *   -Calculates the absolute difference between each pair of weights and sums them up.
         *   -Returns the total net change in weights across all emotions and terms.
         */
        private static double CalculateNetChange(Dictionary<string, Dictionary<string, double>> previousWeights,
            Dictionary<string, Dictionary<string, double>> currentWeights)
        {
            double netChange = 0.0;

            foreach (var emotion in currentWeights.Keys)
            {
                foreach (var term in currentWeights[emotion].Keys)
                {
                    double previousWeight = previousWeights[emotion][term];
                    double currentWeight = currentWeights[emotion][term];

                    netChange += Math.Abs(currentWeight - previousWeight);
                }
            }

            return netChange;
        }

        /*
         * Saves a list of net changes to a specified file.
         *
         * Parameters:
         *   -netChanges: List of net changes to be saved.
         *   -filePath: File path to save the net changes.
         *
         * Implementation Details:
         *   -Formats each net change value with its index as a string.
         *   -Writes the formatted data, consisting of index-value pairs, to the specified file.
         */
        private static void SaveDataToFile(List<double> netChanges, string filePath)
        {
            var lines = netChanges.Select((val, index) => $"{index + 1}, {val}");

            File.WriteAllLines(filePath, lines);
        }
    }
}
