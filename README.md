# Comprehensive AI Agents and Interactive UI
# Project Description

Comprehensive AI agents comprise various tools aimed at enhancing tasks such as internet searches, web scraping, data conversation, and trip planning. The project includes:

- **Multi-Tool Agent**: Capable of navigating between utilities like internet searches, web scraping, and two RAG (Retrieval-Augmented Generation) tools for conversing with data.
- **Agent Orchestration Tool**: Designed specifically for planning trips.
- **Comprehensive RAG Tool**: Equipped with multiple retrieval methods for optimal information retrieval operations.

## Libraries Used:
- Llama-index
- Langchain
- CrewAI
- OpenAI

## Running the Project:

1. Download Ollama using the following command:
    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ```

2. Make the Ollama model setup script within the setup directory executable using:
    ```
    chmod +x <filepath>
    ```

3. Run the Ollama model setup files.

4. Use the following command to run the Ollama server and utilize the models:
    ```
    ollama serve
    ```

5. Open the virtual environment using Poetry shell:
    ```
    poetry shell
    ```

6. Finally, run the Flask app using:
    ```
    python app.py
    ```

Run the deployed app via the link:
    ```
    https://comprehensive-ai-agent.onrender.com
    ```

