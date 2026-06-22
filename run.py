"""Entrypoint: run the chatbot-agent FastAPI server via uvicorn."""

import uvicorn

from agent.config import settings


def main() -> None:
    uvicorn.run(
        "agent.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
