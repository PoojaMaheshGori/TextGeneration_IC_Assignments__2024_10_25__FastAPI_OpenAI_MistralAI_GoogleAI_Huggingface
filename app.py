#region : Import Statements
'''
pip install fastapi uvicorn
uvicorn src.fastAPI_toDo_app:app --reload
'''
import os
import subprocess
import time

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.genAI_online import GenAI_Online

#endregion

class Services:
    
    # fastapi_process = None
    def __init__(self) -> None:
        self.app = FastAPI()  # Create FastAPI instance
        load_dotenv()
        # Register endpoints with FastAPI
        service_GenAI_Online = GenAI_Online()
        
        self.app.add_api_route("/summarize", service_GenAI_Online.summarize, methods=["POST"])
        self.app.add_api_route("/generate", service_GenAI_Online.generate, methods=["POST"])
        self.app.add_api_route("/translate", service_GenAI_Online.translate, methods=["POST"])
        
        # self.app.add_api_route("/summarize", service_GenAI_Online.summarize, methods=["POST"])
        # self.app.add_api_route("/summarize_mistralai", service_GenAI_Online.summarize_mistralai, methods=["POST"])
        # self.app.add_api_route("/summarize_openai", service_GenAI_Online.summarize_openai, methods=["POST"])
        
        # self.app.add_api_route("/generate_mistralai", mistralService.generate_mistralai, methods=["POST"])
        
        return None
    
    def fetch_app(self):
        return self.app
    
    #region : Only used for hosting fast API from other file like streamlit
    def connect_FastAPI(self, port="8000", retry=2):
        global fastapi_process
        fastApi_url = "http://127.0.0.1:" + port
        start_command = ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", port, "--reload"]
        
        if not self.check_server_status(fastApi_url):
            #"""Start the FastAPI server in a subprocess."""
            fastapi_process = subprocess.Popen( start_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, ) 
            time.sleep(3)  # Wait for the FastAPI server to start
    
        return fastApi_url
    
    def check_server_status(self, fastApi_url):
        #"""Check if the FastAPI server is running."""
        for _ in range(10):  # Retry for a few seconds
            try:
                response = requests.get(fastApi_url)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                time.sleep(1)
        return False
    #endregion
    
app = Services().fetch_app()

# # Run the Streamlit app
# if __name__ == "__main__":
#      # Start the FastAPI server
#     fastApi_url = services.connect_FastAPI("9000")