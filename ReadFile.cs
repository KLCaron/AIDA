using Newtonsoft.Json;
using System;
using System.IO;

namespace AIDA
{
    public class ReadFile
    {
        public static T ReadJson<T>(string jsonFilePath)
        {
            try
            {
                string jsonText = File.ReadAllText(jsonFilePath);
                T data = JsonConvert.DeserializeObject<T>(jsonText);
                return data;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {jsonFilePath}: {ex.Message}");
                return default(T);
            }
        }
    }
}