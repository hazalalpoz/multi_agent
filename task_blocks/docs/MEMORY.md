# Memory

## The four kinds, and where each one lives

| Kind | Description | Implementation here |
|---|---|---|
| Working / short-term | The current conversation: everything said, in order | LangGraph `SqliteSaver` checkpointer, keyed by `thread_id`. Complete and chronological, trimmed to the last `HISTORY_MAX_MESSAGES` (12) when passed to a model |
| Episodic | Specific past events, such as "last time we spoke you had just moved" | Partially. Resumable threads hold past episodes, and a memory may record one, but there is no event timeline. The weakest of the four here |
| Semantic | Durable facts about the person, detached from when they were said | `SqliteStore` with a vector index, namespace `("memories", user_id)` |
| Procedural | How to do things | The `agent/` files: `SOUL.md`, the role prompts, and the skills. Versioned in the repository rather than learned at runtime, so a human reviews any change to how the system behaves |

## What was built

Short-term memory is a `SqliteSaver` over `data/checkpoints.sqlite`. A turn
passes in only the new message; earlier ones come from the thread and
`add_messages` appends. The `--thread <id>` flag resumes a conversation. This
was verified: a reply referred correctly to what had been asked one turn
earlier, and conversations survived a container restart from the volume.

Long-term memory is a `SqliteStore` with
`index={"dims": 768, "embed": nomic-embed-text, "fields": ["content"]}`, backed
by `sqlite-vec`. Retrieval is semantic search over the user's namespace with
the latest message as the query, limited to five results. The limit is small
because results go into every system prompt, and a model given twenty
recollections starts reciting them.

Writes are always an agent's explicit decision, through LangMem's
`manage_memory` tool. Nothing extracts facts automatically, for two reasons: a
small local model extracting facts from every turn fills the store with
transient noise such as "is tired today", and silently recording what someone
says in an emotional conversation is the wrong default for this use case.

Scoping is structural. The namespace template `("memories", "{user_id}")` is
filled from the run configuration, so a tool call can only reach the caller's
memories. Scoping at the namespace level rather than by filtering results means
a prompt-injected search cannot cross users. This was verified: a second user
asking the same question received "I don't have a memory of your dog."

Crisis turns have no memory tools at all. The crisis skill forbids storing
anything from such a conversation, and removing the capability is a stronger
guarantee than instructing a model not to use it. This was verified with a turn
saying "remember that I feel this way every November": the store was unchanged.

## Why plain RAG is not the right fit

By plain RAG is meant chunking documents, embedding them, retrieving top-k by
cosine similarity, and placing the result in the prompt. It is the right tool
for answering questions over a corpus, and a poor fit for remembering a person,
for five reasons.

First, stateless chunks have no identity. If someone says in March "I have a
border collie" and in September "we had to put her down", plain RAG holds both
and retrieves both, with no notion that the second supersedes the first. A
memory store needs update and delete, which LangMem's `manage_memory` provides
as first-class actions.

Second, there is no forgetting. A corpus is meant to be permanent; a person's
facts are not. Nothing in plain RAG expresses that something stopped being true
or should expire. The save policy here tries to prevent transient facts being
stored at all, which is prevention rather than a cure.

Third, user scoping is not a primitive. It becomes a metadata filter applied
after retrieval, so correctness depends on remembering to pass the right
filter. Namespacing makes it structural.

Fourth, there is no temporal validity. "Six months into a two-year contract" is
only meaningful with a date attached, and retrieval by similarity has no notion
of when, so a model receives a fact with no way to judge its currency. The
mitigation here is a convention, since the note-taking skill asks for date
stamps on time-bound facts, which is weaker than a system that models time.

Fifth, similar is not relevant. Cosine similarity retrieves what reads like the
query. Asked "should I take this job?", the useful memory may be "hates
commuting", which is related by inference rather than by wording. This cuts
both ways: the host-mode keyword fallback here ranks "dissertation" correctly
but cannot connect "pet" to "border collie" at all, which is the gap embeddings
close. Embeddings narrow the problem without eliminating it.

The distinction is that RAG answers what a corpus says about a topic, while
memory answers what is known about this person that matters now.

## Alternatives compared

| System | Offering | Reason not chosen |
|---|---|---|
| Graphiti | Temporally-aware knowledge graph with entities, relations and validity intervals, resolving contradictions over time | The most compelling alternative and the clear upgrade path. It needs Neo4j, which is another container and another store to back up, and its extraction quality depends on a capable model, which conflicts with local-only extraction on a 1.7B model |
| Zep | Managed memory service with a temporal graph and summarisation | Hosted, so conversation content leaves the host, which is the wrong property here. Self-hosting is possible but heavier than the rest of the stack |
| Mem0 | A dedicated memory layer with extraction, deduplication and decay, behind a simple API | Closest in spirit to what was built. Rejected because it is a second abstraction over a store LangGraph already provides, and it wants an LLM for extraction. Worth revisiting if automatic extraction becomes desirable |
| Letta (MemGPT) | An agent architecture with self-editing memory and paging between context and storage | A whole agent framework rather than a memory component, so adopting it means replacing LangGraph. Its self-editing core memory is a good idea for long relationships |
| Cognee | An ECL pipeline building a semantic graph from documents | Document-oriented. These memories are single sentences, so the pipeline has little to work with |
| LlamaIndex | Mature indexing and retrieval with many index types | Excellent retrieval over corpora, which is the RAG framing above. Its chat memory is thinner than its indexing |
| PageIndex | Hierarchical, reasoning-based retrieval over long documents | Built for long structured documents. A store of one-line facts has no hierarchy to exploit |
| SuperMemory / memU | Hosted memory APIs with cross-session recall | Hosted, so the same objection as Zep, with less control over deletion, which matters when the policy is never to store something |
| Rowboat | A multi-agent builder with memory included | Solves the orchestration problem already solved with LangGraph, so adopting it for memory means adopting its orchestration |
| RAGFlow | Document understanding and RAG serving with strong parsing | Aimed at document ingestion, and no documents are ingested here |
| LangMem + LangGraph Store (chosen) | Memory tools over the store the framework already has, namespaced, with local embeddings and explicit create, update and delete | Native integration, no extra service, no data leaving the host, and the agent decides what is worth keeping |

## Drawbacks of the chosen design

The choice is defensible but not free.

There is no temporal reasoning: no validity intervals and no "as of". Date
stamps are a convention in a prompt, which makes them advisory.

There are no relations. Facts are independent strings, so "partner is named
Sam" and "Sam changed jobs" are unconnected, where a graph would link them.

Extraction quality is the model's. Writes depend on a model judging what is
durable and following the save policy. In observation it saved the right facts
on request, but the policy is instruction-following rather than enforcement.
The one hard guarantee is the crisis case, enforced by withholding the tools.

Retrieval has no reranking. The top five by cosine similarity go straight into
the prompt, with no cross-encoder, no diversity and no relevance threshold, so
an irrelevant memory can crowd out a relevant one.

The store is single-node SQLite, with one writer and one host. This is adequate
for one user and a hard limit for concurrency.

LangMem is pre-1.0, with a young API, and it pulls `langchain-anthropic`
transitively although it is never imported.

Behaviour depends on the environment. The host's CPython cannot load SQLite
extensions, so `sqlite-vec` is unavailable and search falls back to keyword
ranking. This is logged at ERROR on every start and `/health` reports
`degraded`, but it means host-mode memory numbers are not comparable with
deployed ones. Memory recall must be measured in the container.

Finally, a model may ignore what it retrieved. Asked directly "What do I do for
a living?" with the fact in its prompt, both generating models answered that
they did not know, while using the same fact correctly when asked indirectly.
Retrieval works; the model's use of it is unreliable in one specific shape. See
[EVALS.md](EVALS.md) for the measurement and what was tried.

## Improvements, in priority order

1. Fix the retrieved-memory denial. This is the most valuable change, because
   retrieval working is worthless if the model disclaims it. The next step is
   the isolating experiment that the rate limit interrupted: the memories block
   alone, versus with `SOUL.md`, versus with the role prompt, to find which
   instruction the model is privileging. The suspect is `SOUL.md`'s instruction
   never to pretend to remember something it did not store.
2. Reranking and a relevance threshold. Retrieve 20, rerank with a
   cross-encoder, and pass 3 above a score. This would cut the
   irrelevant-memory noise that may contribute to the first item.
3. Temporal metadata as a field rather than a convention. Store `asserted_at`
   and an optional `valid_until`, expose age to the model, and expire what has
   lapsed.
4. A consolidation job to merge duplicates periodically, resolve contradictions
   in favour of the newer fact, and drop what has not been retrieved in months.
   Running offline means it can use a better model than the request path
   affords.
5. Graphiti for relations and time, once a second container is acceptable. Its
   temporal knowledge graph addresses the first two drawbacks together.
6. `PostgresStore` with pgvector, which removes both the single-writer limit
   and the SQLite-extension problem at the cost of another container. It would
   also make host and container behaviour identical, simplifying measurement.
