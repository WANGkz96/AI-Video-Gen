"""Explicit operator recovery on the same GPU; refuses any job with media.

Run only after stopping the old API and its generation child processes.
Keeps the original job id so the existing authenticated polling/tunnel continues.
"""
import argparse
from contextlib import asynccontextmanager

import uvicorn

from backend.app.main import app, job_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        await job_service.restore_empty_job(args.job_id)
        async with original(application):
            yield

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
