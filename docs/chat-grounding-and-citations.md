# Grounding and public citations

`RAGChain.answer()` is the common generation boundary for JSON and SSE in every
reranker mode. SSE waits for the whole result and never forwards provider tokens.
There is one answer-generation call, with no new repair call or retry policy.

The internal answer must retain `[n]` on each factual logical claim. The existing
validator checks marker validity, figures, entity codes, exams, campus scope and
entity/value relationships in the cited sentence, list item or table row. A
source list at the end cannot replace claim attribution. The three whole-answer
safe abstentions remain a separate narrow allowlist; mixed factual refusals must
pass factual grounding and are rejected when unsupported.

Only after successful validation does the chain select used source markers,
build citations from the corresponding original chunk metadata, then remove
inline markers from the public answer. Duplicate retrievals of the same evidence
identity share one citation. Distinct evidence locations from one document retain
their own chunk IDs, locators and excerpts. Structural-only markers do not create
public citations. The public citation `marker` still identifies the original
internal source; subset citations need not be consecutively numbered.

The API persists the same public answer/citations it returns. The frontend renders
Markdown followed by one **Nguồn tham khảo** section. Evidence with matching
document ID, URL, type and publication date shares one source link while keeping
its markers, locations and distinct excerpts. Different known document versions
and sources without an established identity remain separate. Sources contain existing
titles, URLs and provenance; the model cannot create links or a second footer.
An `out_of_scope` response has no citations or source section.

Backend-generated `X-Request-ID` is preserved through the Next proxy, including
grounding/provider errors. Internal `rag_generation` diagnostics contain the
request ID, SHA-256 question hash, selected chunk IDs, marker mapping, response
classification, rejection reason and claim index. They exclude question/answer
content. Provider and persistence error logs retain error classes rather than
SDK/SQL tracebacks that could contain private payloads. Query-rewrite completion
logs also omit query contents.

Grounding remains conservative, based on explicit patterns rather than general
semantic entailment. Unknown aliases, ambiguous relationships or unsupported
generation may still produce `503 GENERATION_UNGROUNDED`. Provider failures keep
their own error codes. SSE buffering increases the time before the first answer
chunk in modes that previously streamed unchecked tokens.

Offline tests use pinned local evidence and fake retrieval/provider responses.
They do not establish live retrieval quality or provider/browser acceptance.
Live checks require an already approved healthy runtime, without starting or
restarting API services or constructing another BGE-M3 instance.

For provider interruptions, a narrow evidence recovery handles only a general
PTIT admission-methods overview. It requires an explicit method count, exactly
that complete numbered list, trusted PTIT publisher metadata, one admission
year, no conflicting campus scope and no conflicting method definitions. It
copies source titles and runs the same validator before returning an answer.
Filtered questions, another institution, missing methods and ambiguous evidence
decline recovery. Authentication, known factual-relationship errors and invalid
markers never take this path. A `missing_citation` rejection may also request
independent recovery: discard the entire model output, copy only the complete
source-supported methods list and validate that new answer. Never guess or
attach citations to the model's uncited claims. `answer_mode="evidence"`
identifies recovery in JSON/SSE and the UI; the original rejection is still
recorded internally. Other failures retain explicit
503/504 error responses and never become invented successful answers.

The frontend preserves the failed question and offers a manual resend for
transient provider errors. It makes no automatic generation retry and keeps the
input available while the retrieval backend remains usable. The proxy translates
legacy upstream 502 and malformed responses into structured 503 failures while
preserving the error code and request correlation.
