using Newtonsoft.Json;
using System;
using System.IO;

namespace AIDA
{
    public static class ReadFile
    {
        public static T ReadJson<T>(string filePath)
        {
            try
            {
                string jsonText = File.ReadAllText(filePath);
                T data = JsonConvert.DeserializeObject<T>(jsonText);
                return data;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {filePath}: {ex.Message}");
                return default(T);
            }
        }
        
        public static T ReadTxt<T>(string filePath)
        {
            T result = default;

            try
            {
                using (StreamReader reader = new StreamReader(filePath))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        result = (T)(object)line;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {filePath}: {ex.Message}");
            }

            return result;
        }
    }
}