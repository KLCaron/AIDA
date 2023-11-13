using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace AIDA
{
    public static class ReadFile
    {
        public static T ReadJson<T>(string filePath)
        {
            try
            {
                string jsonText = File.ReadAllText(filePath);

                if (typeof(T) == typeof(JObject))
                {
                    return (T)(object)JObject.Parse(jsonText);
                }
                else
                {
                    T data = JsonConvert.DeserializeObject<T>(jsonText);
                    return data;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {filePath}: {ex.Message}");
                return default;
            }
        }
        
        public static List<string> ReadTxt(string filePath)
        {
            List<string> result = new List<string>();

            try
            {
                result = File.ReadAllLines(filePath).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {filePath}: {ex.Message}");
            }

            return result;
        }
    }
}