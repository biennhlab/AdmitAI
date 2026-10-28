"""Request correlation without storing private questions or provider output."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import logging
import re
from uuid import uuid4

_request = ContextVar('rag_diagnostics', default=None)
logger = logging.getLogger(__name__)


@contextmanager
def generation_diagnostics(question: str, request_id: str | None = None):
    current = _request.get()
    token = _request.set(current or {
        'request_id': request_id or uuid4().hex,
        'question_hash': hashlib.sha256(question.encode('utf-8')).hexdigest(),
    })
    try:
        yield
    finally:
        _request.reset(token)


def record_generation(sources, answer, classification, *, error=None):
    mapping = {str(index): source.prompt_chunk.chunk_id for index, source in enumerate(sources, 1)}
    markers = list(dict.fromkeys(re.findall(r'\[(\d+)\]', answer))) if isinstance(answer, str) else []
    fields = {
        **(_request.get() or {}), 'provider_response_class': classification,
        'outcome': 'rejected' if error else 'accepted',
        'reason': getattr(error, 'reason', getattr(error, 'code', None)),
        'claim_index': getattr(error, 'claim_index', None),
        'error_class': type(error).__name__ if error else None,
        'selected_source_ids': list(mapping.values()),
        'citation_mapping': mapping, 'answer_markers': markers,
    }
    emit = logger.warning if error else logger.info
    emit('rag_generation %s', json.dumps(fields, ensure_ascii=True, sort_keys=True))
