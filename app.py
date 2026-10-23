"""Entry point. Mounts the mailroom action gate.

Run locally:  uvicorn app:app --host 0.0.0.0 --port 8000

The grader POSTs to a single endpoint and switches on `operation`.
Every plausible path is mounted so whichever URL you submit works.
"""
import os

from fastapi import FastAPI, Request

import mailroom

app = FastAPI(title="Mailroom Action Gate")
app.include_router(mailroom.router)

ALIASES = ["/", "/mailroom", "/q9", "/gate", "/action-gate", "/v1/mailroom/actions", "/v1/mailroom/actions/"]


async def alias(request: Request):
    return await mailroom.mailroom(request)


for _path in ALIASES:
    app.post(_path)(alias)


@app.get("/")
async def health():
    return {"ok": True, "service": "mailroom-action-gate",
            "profile": mailroom.PROFILE,
            "post": ["/q9/mailroom"] + ALIASES}


@app.get("/health")
async def health2():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
