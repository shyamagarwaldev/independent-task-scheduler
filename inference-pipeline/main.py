import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from cache import cache
from scheduler import start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # start background scoring loop on startup
    start_scheduler()
    yield
    # nothing to clean up, daemon thread dies with process


app = FastAPI(
    title="Congestion Inference Pipeline",
    description="Scores Kubernetes nodes for the congestion-aware scheduler",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "scored_nodes": len(cache.get_all())}


@app.get("/scores")
def get_scores():
    scores = cache.get_all()
    if not scores:
        # return empty dict, not 503 — scheduler handles missing nodes gracefully
        return JSONResponse(content={})
    return JSONResponse(content=scores)


@app.get("/scores/{node_name}")
def get_score(node_name: str):
    score = cache.get(node_name)
    if score is None:
        raise HTTPException(status_code=404, detail=f"no score for node {node_name}")
    return JSONResponse(content=score)