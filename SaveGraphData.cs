using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace AIDA
{
    public static class SaveGraphData
    {
        public static void SaveNetChangesToFile(string basePath, string outputFilePath)
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

            SaveDataToFile(netChanges, outputFilePath);
        }

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

        private static void SaveDataToFile(List<double> netChanges, string filePath)
        {
            var lines = netChanges.Select((val, index) => $"{index + 1}, {val}");

            File.WriteAllLines(filePath, lines);
        }
    }
}
