import os

class OllamaConfig:
    def __init__(self):
        self.OLLAMA_HOST = os.getenv('OLLAMA_HOST')
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
        self.LLM_MODEL = os.getenv('LLM_MODEL')
        self.MODEL_TEMPERATURE = 0