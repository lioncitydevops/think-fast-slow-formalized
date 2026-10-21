import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..core.types import Query, DecisionBand
from ..config import config
from ..system1.jev_client import JevClient
from ..system2.qwen_client import QwenClient
from ..patterns import (
    create_p1_cascade,
    create_p2_budgeted,
    create_p3_probe,
    create_p4_verifier,
    create_p5_reconcile,
)
from ..patterns.controller import DualProcessController
from ..domains.applications import ALL_DOMAINS
from ..evaluation.harness import EvaluationHarness
from ..evaluation.replications import (
    replicate_table_3,
    replicate_table_5,
    replicate_table_6,
    replicate_table_9,
    generate_figure_1_data,
    generate_figure_2_data,
    generate_figure_4_data,
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Thinking, Fast and Slow, Formalized API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

class QueryRequest(BaseModel):
    text: str
    question: str
    options: List[str]
    pattern: str = "P3"
    tau: float = 0.88
    use_mock: bool = False
    task: Optional[str] = "general"
    probe: bool = True
    reconcile: bool = False
    verify: bool = False

@app.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Check connectivity to TypeSafe Jev and local Qwen3-8B engine."""
    has_typesafe_key = bool(config.typesafe_api_key and len(config.typesafe_api_key) > 10)

    # Check local Qwen server
    qwen_online = False
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.qwen_api_base, api_key="none", timeout=1.5)
        models = client.models.list()
        qwen_online = True
    except Exception:
        qwen_online = False

    return {
        "system1": {
            "name": "TypeSafe Jev (System 1)",
            "configured": has_typesafe_key,
            "model": config.typesafe_default_model,
            "status": "Online (Live API Key Configured)" if has_typesafe_key else "Offline / Emulator Mode",
        },
        "system2": {
            "name": "Qwen3-8B (System 2)",
            "endpoint": config.qwen_api_base,
            "status": "Online (GPU Accelerated on RTX 3070)" if qwen_online else "Offline / Emulator Mode",
            "online": qwen_online,
        },
        "config": {
            "default_tau": 0.88,
            "default_r": config.r,
            "default_beta": config.beta,
            "default_lambda_beta": config.lambda_beta,
        },
    }

@app.get("/api/domains")
def get_domains() -> Dict[str, Any]:
    """Return all Table 4 domain application profiles."""
    serialized = {}
    for key, dom in ALL_DOMAINS.items():
        serialized[key] = {
            "name": dom.domain_name,
            "system1_questions": dom.system1_questions,
            "system2_task": dom.system2_task,
            "costly_error": dom.costly_error,
            "fallback_action": dom.fallback_action,
            "risk_to_monitor": dom.risk_to_monitor,
            "sample_queries": [
                {
                    "id": q.id,
                    "text": q.text,
                    "question": q.question,
                    "options": q.options,
                    "gold_answer": q.gold_answer,
                }
                for q in dom.sample_queries
            ],
        }
    return serialized

@app.post("/api/query")
def execute_query(req: QueryRequest) -> Dict[str, Any]:
    """Execute dual-process controller pipeline on user query."""
    query = Query(
        id=f"web_{int(time.time()*1000)}",
        text=req.text,
        question=req.question,
        options=req.options,
        task=req.task,
    )

    jev = JevClient(use_mock=req.use_mock)
    qwen = QwenClient(use_mock=req.use_mock)

    if req.pattern == "P1":
        controller = create_p1_cascade(jev, qwen, tau=req.tau)
    elif req.pattern == "P2":
        controller = create_p2_budgeted(jev, qwen, tau=req.tau)
    elif req.pattern == "P3":
        controller = create_p3_probe(jev, qwen, tau=req.tau)
    elif req.pattern == "P4":
        controller = create_p4_verifier(jev, qwen, tau=req.tau)
    elif req.pattern == "P5":
        controller = create_p5_reconcile(jev, qwen, tau=req.tau)
    else:
        controller = DualProcessController(
            jev_client=jev,
            qwen_client=qwen,
            tau=req.tau,
            probe=req.probe,
            reconcile=req.reconcile,
            verify=req.verify,
        )

    res = controller.execute(query)

    qwen_info = None
    if res.qwen:
        qwen_info = {
            "action": res.qwen.action,
            "reasoning_trace": res.qwen.reasoning_trace,
            "tokens_used": res.qwen.tokens_used,
            "latency": round(res.qwen.latency, 2),
            "mode": res.qwen.mode,
        }

    return {
        "action": res.action,
        "band": res.band.value,
        "mode": res.mode,
        "tokens_spent": res.tokens_spent,
        "total_latency": round(res.total_latency, 2),
        "conformal_set": res.conformal_set,
        "notes": res.notes,
        "system1": {
            "action": res.jev.action,
            "confidence": round(res.jev.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in res.jev.probabilities.items()},
            "latency": round(res.jev.latency, 3),
        },
        "system2": qwen_info,
    }

@app.get("/api/replications")
def get_replications() -> Dict[str, Any]:
    """Return Table 6, Table 9, Figure 1, Figure 2, and Figure 4 data."""
    t6 = replicate_table_6()
    t9 = replicate_table_9(n_samples=5000)
    return {
        "table_3": t6,
        "table_6": t6,
        "table_5": t9,
        "table_9": t9,
        "figure_1": generate_figure_1_data(),
        "figure_2": generate_figure_2_data(),
        "figure_4": generate_figure_4_data(),
    }

@app.post("/api/evaluate")
def run_evaluation(samples: int = 20) -> Dict[str, Any]:
    """Execute Section 8 verified evaluation suite."""
    harness = EvaluationHarness(use_mock=True)
    return harness.run_full_suite(max_samples_per_task=samples)

# Static files mount
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Thinking, Fast and Slow, Formalized API active."}
