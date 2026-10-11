import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loaders import load_config
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException


# ---------------------------------------------------------------------------
# Required API keys per provider
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = {
    "evren": ["EVREN_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
}


class ApiKeyManager:
    """
    Resolves API keys from:
      1. A JSON blob stored in the 'apikeyliveclass' env var (ECS / secrets manager)
      2. Individual environment variables as fallback
    """

    def __init__(self, required_keys: list[str]):
        self.api_keys: dict[str, str] = {}

        # --- Try JSON blob first (production secrets) -----------------------
        raw = os.getenv("apikeyliveclass")
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("apikeyliveclass is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API keys from ECS secret (apikeyliveclass)")
            except Exception as e:
                log.warning("Failed to parse apikeyliveclass as JSON", error=str(e))

        # --- Fallback: individual env vars -----------------------------------
        for key in required_keys:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # --- Final validation ------------------------------------------------
        missing = [k for k in required_keys if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info(
            "API keys loaded",
            keys={k: v[:6] + "..." for k, v in self.api_keys.items()},
        )

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key '{key}' is missing")
        return val


class ModelLoader:
    """
    Loads the embedding model and LLM based on config.yaml and environment
    variables.

    Providers
    ---------
    LLM      : evren  -> ChatOpenAI (OpenAI-compatible endpoint, deepseek-v4-flash)
    Embedding: google -> GoogleGenerativeAIEmbeddings (gemini-embedding-001)
    """

    def __init__(self):
        # Load .env only when running locally
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

        # Determine which API keys are needed based on configured providers
        llm_provider: str = self.config["llm"]["provider"]
        emb_provider: str = self.config["embedding_model"]["provider"]

        required_keys: list[str] = []
        for provider in {llm_provider, emb_provider}:
            required_keys.extend(_REQUIRED_KEYS.get(provider, []))

        self.api_key_mgr = ApiKeyManager(required_keys=list(set(required_keys)))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def load_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """
        Returns a GoogleGenerativeAIEmbeddings instance.
        Model is read from config.yaml -> embedding_model.model_name
        """
        try:
            emb_cfg = self.config["embedding_model"]
            model_name: str = emb_cfg["model_name"]   # e.g. models/gemini-embedding-001
            log.info("Loading embedding model", model=model_name, provider="google")

            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),  # type: ignore[arg-type]
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys) from e

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    def load_llm(self) -> ChatOpenAI:
        """
        Returns a ChatOpenAI instance pointed at the Evren API
        (OpenAI-compatible endpoint).

        Config keys consumed from config.yaml -> llm:
          provider          : "evren"
          model_name        : e.g. "deepseek-v4-flash"
          temperature       : default 0
          max_output_tokens : default 2048
          timeout           : default 60
          max_retries       : default 3
        """
        try:
            llm_cfg = self.config["llm"]
            provider: str = llm_cfg["provider"]
            model_name: str = llm_cfg.get(
                "model_name",
                os.getenv("EVREN_MODEL", "deepseek-v4-flash"),
            )
            temperature: float = llm_cfg.get("temperature", 0)
            max_tokens: int = llm_cfg.get("max_output_tokens", 2048)
            timeout: int = llm_cfg.get("timeout", 60)
            max_retries: int = llm_cfg.get("max_retries", 3)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "evren":
                base_url: str = os.getenv(
                    "EVREN_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1"
                )
                return ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get("EVREN_API_KEY"),  # type: ignore[arg-type]
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    max_retries=max_retries,
                )

            # Extendable: add more providers here (openai, groq, etc.)
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: '{provider}'")

        except (ValueError, KeyError):
            raise
        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Failed to load LLM", sys) from e


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding model loaded: {embeddings.model}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding vector length: {len(result)}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM loaded: {llm.model_name}")
    response = llm.invoke("Hello, how are you?")
    print(f"LLM response: {response.content}")