"""Bounded policy executor and outer-training-only, diagnosis-free retrieval.

No LLM controller or paper fault results are implied by this infrastructure.
"""
from dataclasses import dataclass
import time

import numpy as np


TOOLS = ('inspect_quality', 'get_region_evidence', 'retrieve_training_cases', 'check_summary')


class TrainingRetriever:
    def __init__(self, subject_ids, embeddings, records, allowed_training_ids):
        if not set(subject_ids).issubset(set(allowed_training_ids)):
            raise ValueError("Retrieval index contains a non-training subject")
        vectors = np.asarray(embeddings, dtype=float)
        if len(subject_ids) != len(vectors) or len(records) != len(vectors) or not np.isfinite(vectors).all():
            raise ValueError("Invalid retrieval index")
        self.ids = list(subject_ids)
        self.vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        # Only evidence records are accepted; diagnosis/label payloads are rejected.
        allowed = {'id', 'region', 'laterality', 'modality', 'value', 'unit', 'source', 'quality'}
        if any(set(record) - allowed for case in records for record in case):
            raise ValueError("Retrieval payload must contain diagnosis-free evidence records only")
        self.records = records

    def search(self, embedding, k=5, exclude_subject=None):
        query = np.asarray(embedding, dtype=float)
        if not np.isfinite(query).all() or k < 1:
            raise ValueError("Expected finite query and positive k")
        score = self.vectors @ (query / max(np.linalg.norm(query), 1e-12))
        order = [i for i in np.argsort(-score) if self.ids[i] != exclude_subject][:k]
        return [{'subject_id': self.ids[i], 'similarity': float(score[i]), 'evidence': self.records[i]} for i in order]


@dataclass
class PolicyResult:
    trace: list
    calls: int
    latency_seconds: float
    stop_reason: str


def execute_policy(tools, choose=None, fixed=False, budget=6):
    """Each actual attempted tool invocation counts; controller reasoning does not.

    choose(trace) returns a permitted tool name or None. State is local per call.
    Tool callables take no arguments and capture the current case in a closure.
    """
    if budget < 0 or set(tools) != set(TOOLS) or (not fixed and choose is None):
        raise ValueError("Supply four tools, a valid budget, and a policy")
    trace, started = [], time.perf_counter()
    while len(trace) < budget:
        name = TOOLS[len(trace)] if fixed and len(trace) < len(TOOLS) else None if fixed else choose(list(trace))
        if name is None:
            return PolicyResult(trace, len(trace), time.perf_counter() - started, 'policy_stop')
        if name not in tools:
            return PolicyResult(trace, len(trace), time.perf_counter() - started, 'invalid_action')
        try:
            value = tools[name]()
            trace.append({'tool': name, 'result': value, 'error': None})
        except Exception as error:
            trace.append({'tool': name, 'result': None, 'error': type(error).__name__})
    return PolicyResult(trace, len(trace), time.perf_counter() - started, 'budget_exhausted')
