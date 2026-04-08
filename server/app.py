"""OpenEnv server entry point — re-exports the FastAPI app."""
import uvicorn
from app.main import app  # noqa: F401


def main():
    """Start the SREBench server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
