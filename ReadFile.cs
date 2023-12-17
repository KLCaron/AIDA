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
        /*
         * Reads JSON data from a file and deserializes it into the specified type.
         *
         * Parameters:
         *   -filePath: File path to the JSON file.
         *
         * Returns:
         *   -T: Deserialized object of type T from the JSON data.
         *
         * Implementation Details:
         *   -Reads the contents of the JSON file located at the provided file path.
         *   -Deserializes the JSON data into the specified type T.
         *   -Handles exceptions that might occur during file reading or deserialization.
         */
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
        
        /*
         * Reads text content from a file and returns it as a list of strings.
         *
         * Parameters:
         *   -filePath: File path to the text file.
         *
         * Returns:
         *   -List<string>: Text content read from the file as a list of strings.
         *
         * Implementation Details:
         *   -Reads the text content from the file located at the provided file path.
         *   -Converts the text content into a list of strings, each representing a line.
         *   -Handles exceptions that might occur during file reading.
         */
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