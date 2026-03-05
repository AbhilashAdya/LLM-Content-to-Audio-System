from app.ingestion.run_ingestion import run_ingestion
from app.assistant.cli_assistant import CLIAssistant
from app.memory.vector_store import VectorStore


def main():
    store = VectorStore()
    run_ingestion(store)
    assistant = CLIAssistant(store)
    assistant.start()


if __name__ == "__main__":
    main()
