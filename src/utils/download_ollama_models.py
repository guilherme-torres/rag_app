import json
import requests
from tqdm import tqdm
from src.config.ollama_config import OllamaConfig

class DownloadOllamaModels:

    def __init__(self, config: OllamaConfig):
        self.config = config

    def execute(self):
        models = (
            self.config.EMBEDDING_MODEL,
            self.config.LLM_MODEL
        )

        for model in models:
            print(f"\nDownloading model: {model}")
            try:
                with requests.post(
                    f"{self.config.OLLAMA_HOST}/api/pull",
                    json={"name": model},
                    stream=True
                ) as response:
                    if response.status_code != 200:
                        print(f"Error: {response.status_code} - {response.text}")
                        continue

                    pbar = None
                    total = None

                    for line in response.iter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            print(f"Invalid response: {line}")
                            continue

                        status = data.get("status")
                        completed = data.get("completed")
                        total = data.get("total")

                        if total and pbar is None:
                            pbar = tqdm(total=total, unit='B', unit_scale=True)
                        
                        if completed and pbar:
                            pbar.n = completed
                            pbar.refresh()

                        if status:
                            print(f"  > {status}")

                    if pbar:
                        pbar.n = pbar.total
                        pbar.refresh()
                        pbar.close()
                        print("Done.")

            except requests.exceptions.RequestException as e:
                print(f"Unable to connect to ollama: {e}")