using Newtonsoft.Json;

namespace AIDA 
{ 
    public class TrainingData //meant to get/set the text and emotions lines in our JSON
    {
        [JsonProperty("text")] //because C# apparently likes capitalized variables
        public string Text { get; set; } //auto-implemented properties are mind blowing tbqh
        [JsonProperty("emotions")]
        public string Emotions { get; set; }
    }
}