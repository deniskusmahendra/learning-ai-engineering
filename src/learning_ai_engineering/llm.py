from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

@lru_cache
def get_groq_client() -> Groq:
    return Groq()