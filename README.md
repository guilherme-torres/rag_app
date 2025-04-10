## Instruções

Primeiramente certifique-se de ter o docker instalado na sua máquina. Se estiver usando Windows, clone o projeto dentro do WSL para executar corretamente o script de instalação dos modelos de linguagem.
### 1. Configuração do Ollama
   1. crie um arquivo `.env` na raiz do projeto e copie as variáveis do arquivo `.env.example` para ele. Preencha as variáveis `EMBEDDING_MODEL` e `LLM_MODEL` com os modelos de sua preferência (consulte no site oficial do [ollama](https://ollama.com/search).
   2. Abra o terminal na raiz do projeto e execute `docker compose -f ollama/docker-compose-cpu.yaml up -d`
   3. edite a variável `models` do arquivo `ollama/download_models.sh` e insira os modelos que você escolheu. Após isso execute os seguintes comandos no terminal:
```
chmod +x ollama/download_models.sh
./ollama/download_models.sh
```