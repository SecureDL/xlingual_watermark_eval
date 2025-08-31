
import os
import time
import anthropic
from exceptions.exceptions import AnthropicModelConfigurationError

api_key = os.environ.get("ANTHROPIC_API_KEY")

class AnthropicAPI:
    """API class for Anthropic API."""
    def __init__(self, model, temperature, system_content):
        """
            Initialize Anthropic API with model, temperature, and system content.

            Parameters:
                model (str): Model name for Anthropic API.
                temperature (float): Temperature value for Anthropic API.
                system_content (str): System content for Anthropic API.
        """

        self.model = model
        self.temperature = temperature
        self.system_content = system_content
        self.client = anthropic.Anthropic(api_key="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",)
        

        # List of supported models
        supported_models = ['claude-3-5-haiku-20241022']

        # Check if the provided model is within the supported models
        if self.model not in supported_models:
            raise AnthropicModelConfigurationError(f"Unsupported model '{self.model}'. Supported models are {supported_models}.")

    def get_result_from_sonnet(self, query):
        """get result from claude-3-5-haiku-20241022 model."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=5000,
            temperature=self.temperature,
            system=self.system_content,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                },
            ]
        )
        return response

    def get_result(self, query):
        """get result from Anthropic API."""
        result = None
        num_attempts = 0
        while result is None:
            try:
                result = self.get_result_from_sonnet(query)
            except Exception as e:
                if num_attempts == 5:
                    print(f"Attempt {num_attempts} failed, breaking...")
                    return query # return the original text if the API fails
                print(f"OpenAI API error: {str(e)}")
                num_attempts += 1
                print(f"Attempt {num_attempts} failed, retrying in 5 seconds...")
                time.sleep(5)
        return result.content[0].text


if __name__ == "__main__":
    system_prompt = f"""
You are a professional translator tasked with translating a given text from one language to another. 
Your goal is to provide an accurate and natural-sounding translation that preserves the meaning, tone, and style of the original text.
\n\nThe source language is English, and the target language for translation is Arabic.
\n\nPlease provide your final translation within <translated_text> tags. 
Do not include the original text or any other commentary outside of these tags.
\n\n<translated_text>\n[Your translation goes here]\n</translated_text>
"""
    anthropic_util = AnthropicAPI('claude-3-5-haiku-20241022', 0.2, system_prompt)
    text = "Whoever gets him, they'll be getting a good one,\" David Montgomery said. INDIANAPOLIS \u2014 Hakeem Butler has been surrounded by some of the best wide receivers on the planet this month, so it's only natural he'd get a touch of star-struck when he got a chance to meet Randy Moss, a Hall of Famer and someone Butler idolized growing up and still considers an inspiration to this day.\n\"It's crazy to see someone you've watched on TV and seen on YouTube and then be standing right there and you're like, 'Man, he's just a regular dude,'\" Butler, a 6-foot 5, 227-pound receiver from Iowa State, said Thursday at the NFL Scouts' Luncheon, which kicks off the week of the NFL Scouts' Breakfast and the NFL Scouts' Luncheon at the JW Marriott Indianapolis, which takes place Sunday and Monday, Feb. 1 and 2, at Lucas Oil Stadium, 1 Lucas Oil Stadium Dr., Indianapolis, IN 46225, and which is open to all NFL Scouts and team personnel and others with a pass from the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is sponsored by the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is organized by the NFL Scouts' Breakfast and the NFL Scouts' Luncheon, and which is hosted by the"
    result = anthropic_util.get_result(text)
    print(result)