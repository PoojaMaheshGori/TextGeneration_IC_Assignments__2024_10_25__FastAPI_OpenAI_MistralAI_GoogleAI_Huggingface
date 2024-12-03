import os

import google.generativeai as genai
import openai
import requests
from dotenv import load_dotenv
from mistralai import Mistral
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from enum import Enum

# from pydantic import BaseModel

class GenAI_Online:
    
    def __init__(self) -> None:
        # Initialize the Mistral client
        self.api_key_mistralai = os.getenv("MISTRAL_API_KEY")
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_googleai = os.getenv("GOOGLE_API_KEY")
        self.api_key_huggingface = os.getenv("HUGGING_FACE_API_KEY")
        
        # Set the API keys
        genai.configure(api_key=self.api_key_googleai)
        openai.api_key = self.api_key_openai
        
        # Set up the Google Generative AI client
        self.client_openai = openai.OpenAI()
        self.client_mistral = Mistral(api_key=self.api_key_mistralai)
        
    class ModelType(Enum):
        MISTRAL = "MISTRAL"
        OPENAI = "OPENAI"
        GEMINI = "GEMINI"   
        ALL = "ALL" 
        huggingface_MISTRAL = "huggingface_MISTRAL"
        
    def summarize(self, story: str, model:ModelType=ModelType.OPENAI, temperature:float=0.7, max_tokens:int=150):
        
        response_mistral = ""
        response_openai = ""
        response_googleai = ""
        response_huggingface = ""
        
        self.ModelType.huggingface_MISTRAL
        
        prompt = f"Please summarize the following text:\n\n{story}"
        messages=[
                    {"role": "user", "content": prompt}
                ]
        if model == self.ModelType.MISTRAL | model == self.ModelType.ALL :
            response_mistral = self.generate_response_mistralai("mistral-large-latest", temperature, max_tokens, messages)
        if model == self.ModelType.OPENAI | model == self.ModelType.ALL :
            response_openai = self.generate_response_openai("gpt-3.5-turbo", temperature, max_tokens, messages)
        if model == self.ModelType.GEMINI | model == "ALL" :
            response_googleai = self.generate_response_googleai("gemini-1.5-flash", prompt)
        if model == self.ModelType.huggingface_MISTRAL | model == self.ModelType.ALL :    
            response_huggingface = self.generate_response_huggingface("mistralai/Mistral-7B-v0.1", temperature, max_tokens, prompt)
        
        # Return the original story and its responses
        return {"original story": story, 
                "response_mistral": response_mistral, 
                "response_openai": response_openai, 
                "response_googleai" : response_googleai,
                "response_huggingface (Mistral-7B)": response_huggingface
                }
    
    def generate(self, title: str, model:str="", temperature:float=0.7, max_tokens:int=150):
        
        prompt = f"Write a very short story about a {title}"
        messages=[
                    {"role": "user", "content": prompt}
                ]
        response_mistral = self.generate_response_mistralai("mistral-large-latest", temperature, max_tokens, messages)
        response_openai = self.generate_response_openai("gpt-3.5-turbo", temperature, max_tokens, messages)
        response_googleai = self.generate_response_googleai("gemini-1.5-flash", prompt)
        
        # Return the original prompt and its responses
        return {"title": title, "response_mistral": response_mistral, "response_openai": response_openai, "response_googleai" : response_googleai }
    
    def translate(self, text: str, source_language: str, target_language: str, model:str="", temperature:float=0.7, max_tokens:int=150):
        
        prompt = f"Translate the following text from {source_language} to {target_language}: \"{text}\""
        messages=[
                    {"role": "user", "content": prompt}
                ]
        response_mistral = self.generate_response_mistralai("mistral-large-latest", temperature, max_tokens, messages)
        response_openai = self.generate_response_openai("gpt-3.5-turbo", temperature, max_tokens, messages)
        response_googleai = self.generate_response_googleai("gemini-1.5-flash", prompt)
        
        # Return the original prompt and its responses
        return {"original text": text, "response_mistral": response_mistral, "response_openai": response_openai, "response_googleai" : response_googleai }

    
    def summarize_mistralai(self, story: str, model="mistral-large-latest", temperature=0.7, max_tokens=150):
    # Get the story from the request

        try:
            
            messages=[
                    {"role": "user", "content": f"Please summarize the following text:\n\n{story}"}
                ]
            summary = self.generate_response_mistralai(model, temperature, max_tokens, messages)
            
            # Return the original story and its summary
            return {"story": story, "summary": summary}

        except Exception as e:
            # Handle errors and return a meaningful message
            return {"error": str(e)}

    def summarize_openai(self, story: str, model:str="gpt-3.5-turbo", temperature:float=0.7, max_tokens:int=150):
    # Get the story from the request

        try:
            messages=[
                    {"role": "user", "content": f"Please summarize the following text:\n\n{story}"}
                ]
            summary = self.generate_response_openai(model, temperature, max_tokens, messages)

            # Return the original story and its summary
            return {"story": story, "summary": summary}

        except Exception as e:
            # Handle errors and return a meaningful message
            return {"error": str(e)}

    #region : Online APIs implementation for text generation
    
    def generate_response_openai(self, model, temperature, max_tokens, messages):
        # Generate a response using OpenAI's API
        response = self.client_openai.chat.completions.create(
                model=model,  # Specify the model
                messages = messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract the generated summary
        return response.choices[0].message.content.strip()
    
    def generate_response_mistralai(self, model, temperature, max_tokens, messages):
        # Generate a response using OpenAI's API
        response = self.client_mistral.chat.complete(
                model=model,  # Specify the model
                messages = messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract the generated summary
        return response.choices[0].message.content.strip()

    def generate_response_googleai(self, model, prompt):
        # Generate story using the provided title
        
        messages = [{'role':'user', 'parts': [prompt]}]
        model = genai.GenerativeModel(model)
        response = model.generate_content(messages) # "Hello, how can I help"
        
        return response.text.strip()
        #return response.candidates[0].content
        
        # messages.append(response.candidates[0].content)
        # messages.append({'role':'user', 'parts': ['How does quantum physics work?']})
        # response = model.generate_content(messages)

        
        
         # Prepare prompt from the messages
        # prompt = "\n".join([msg["content"] for msg in request.messages])
        model = genai.GenerativeModel(model)
        # Generate response using Google Generative AI's API
        response = genai.gener .generate_content(
            model=model,
            prompt=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        response = model.ge     .generate_content(f"Write a very short story about a {title}")

        # Extract the generated content
        return response.text.strip()

    def generate_response_huggingface(self, model_name, temperature, max_tokens, prompt):
        
        model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

        # Initialize the text generation pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Generate a story using Mistral-7B-v0.1
        response = generator(
            prompt,
            max_length=max_tokens,  # Maximum length of the generated story
            temperature=temperature,  # Controls creativity/randomness
            num_return_sequences=1  # Generate a single story
        )

        # Extract the generated story
        return response[0]["generated_text"]
    
    def generate_response_huggingface_online(self, model_name, temperature, max_tokens, prompt):
        
        
        
        # Prepare the payload for the API
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        HF_API_URL = "https://api-inference.huggingface.co/models/" +  model_name # Mistral-7B-v0.1 model

        # Prepare the headers
        headers = { "Authorization": f"Bearer {self.api_key_huggingface}"  }

        try:
            # Send the request to the Hugging Face Inference API
            response = requests.post(HF_API_URL, json=payload, headers=headers)
            response.raise_for_status()

            # Extract the generated story
            return response.json()[0]["generated_text"]
        except Exception as e:
            # Handle errors and return a meaningful message
            return {"error": str(e)}
        
    #endregion 
        

# Run the application if executed as the main program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("summarization:app", host="127.0.0.1", port=9001, reload=True)
    
    