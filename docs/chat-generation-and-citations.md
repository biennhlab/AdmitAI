# Answer generation and citations

`RAGChain.answer()` is the shared response path for JSON and SSE in every
reranker mode. It retrieves context, calls the provider once, renders the result
and returns it. SSE presents the complete answer and citation metadata using
the existing `chunk` and `metadata` event schema.

The automatic answer verifier has been removed. There is no check of generated
figures, entity/value relationships, polarity, modality, units, campus scope,
textual overlap or per-claim citation support. Answers are not rejected for
missing source references or for disagreeing with retrieved text. Source guidance
in the generation prompt remains a provider instruction, not an application-side
content validation step.

The provider may return plain text or a structured JSON response:

```json
{"status":"answer","blocks":[{"kind":"paragraph","text":"Answer text.","source_ids":[1]}]}
```

Paragraphs, bullets, numbered items, headings and table rows are rendered as
Markdown. `source_ids` is optional. Valid reference IDs create internal markers;
missing, duplicate, out-of-range or unusable IDs do not reject the text. Existing
markers in ordinary text also map to available source metadata when possible.
An ID establishes a reference, not proof that the source supports the answer.
References are not invented for uncited text.

The presentation layer removes internal numeric markers from the public answer
and includes the referenced sources in citation metadata. Excerpts are selected
for display; this selection does not verify the associated claim.

Only response-format checks remain: empty/non-text responses, malformed JSON,
unrenderable block shapes and invalid structured formatting raise
`GenerationFormatError` with code `INVALID_GENERATION`. The API returns HTTP 502
without exposing provider payloads. The frontend proxy retains its existing
normalization of upstream 502 responses to HTTP 503 while preserving that code.
Provider authentication, timeout, connection, rate-limit and retrieval errors
remain separately classified.

`{"status":"abstain","blocks":[]}` and recognized legacy refusal wording
produce the normal fallback response with no citations. Other ordinary text is
returned without the former content rejection gate.

Retrieval behavior is unchanged: missing evidence may produce a fallback, query
scope rules select source candidates, and complete catalog/profile questions can
use deterministic source-table extraction. A provider failure can still recover
through that extractive path when complete source data is available. Generated
text with missing citations is returned directly, without a verification retry
or replacement answer.

Tests cover JSON/SSE response and persistence parity, optional source references,
format errors, provider failures, retrieval fallback and citation presentation:

```powershell
python -m pytest tests/test_generation_output.py tests/test_structured_answer.py tests/test_api_resilience.py tests/test_citation_presentation.py -q
```
