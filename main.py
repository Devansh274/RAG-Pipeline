"""Single entry point:   python main.py

Run from inside this folder so ./data/ and ./chroma_db/ resolve correctly."""

from dotenv import load_dotenv

load_dotenv()  # load API keys from .env BEFORE any client is created

from pipeline import run

if __name__ == "__main__":
    run()
