# Vector-Searchable Gmail Inbox

A single Python CLI that indexes Gmail (IMAP, app password) into a local
SQLite store and searches it **hybrid**: dense cosine similarity over
**EmbeddingGemma 2** vectors *plus* BM25 keyword search over a SQLite FTS5
index, fused with Reciprocal Rank Fusion. Bodies, metadata (sender / To / Cc /
attachment extensions), attachments (images, PDFs, Office files, optionally
audio) and thread context (prior mails) are all indexed and searchable.

The same index also serves a **mailbox-shaped web UI** (`python mailbox.py gui`):
search, reading pane and inbox browse in the browser — search *with* a picture,
pasted or dropped into the search bar, included — and your own pictures shown from
previews cut while indexing, so a thumbnail is read off disk and Gmail is not
awoken for it. On `127.0.0.1`, with one local Python process, no build step, and
nothing fetched from a CDN.

Each source text is indexed at **three resolutions** — `small` (320 chars),
`large` (1200) and `xlarge` (the whole message) — so a quote inside a long
thread and the gist of that thread are both findable, without a second index.

**No LLM is required.** Documents are embedded as `title: <subject> | text:
<content>` and queries as `task: search result | query: <what you typed>` — the
prompt shape EmbeddingGemma 2 was trained on, which is why the two sides are
not interchangeable and `--profile legacy` keeps the old `Document:` /
`SearchQuery:` shape together. Cosine similarity does the rest; EmbeddingGemma
2's shared 768-dim space handles text and images natively. Keyword hits come
from SQLite's own BM25, not from a model. An LLM would only be an optional
extra (query rewriting, reranking, answer generation).

## Setup

```bash
pip install -r requirements.txt
copy config.example.json config.json   # then fill in your values
```

Fill in `config.json`:

| key | meaning |
|---|---|
| `gmail_user` / `gmail_app_password` | Gmail account + 16-char app password |
| `gmail_account_index` | the `N` of `mail.google.com/mail/u/N/` for this mailbox (default `0`). A second or work account signed into the same browser sits at `u/1`, `u/2`…, and a Gmail link only opens in its own slot — from another one the same message id is different mail or nothing. Change it and every link follows, stored ones included: no re-index |
| `embed_endpoint` | your EmbeddingGemma 2 endpoint: `…/v1/embeddings` (OpenAI shape — OpenAI, vLLM, TEI, llama.cpp) or Ollama's `http://host:11434/api/embed` (native shape) — both are read |
| `embed_api_key` | Bearer key for the endpoint |
| `embed_model` | model id sent as `"model"` (required by hosts such as Ollama); leave `""` to omit it |
| `embed_timeout` | seconds per endpoint request (default 120; lower it to fail fast when the host is down) |
| `folders` | IMAP folders to index (default `["INBOX"]`) |
| `fetch_batch_size` | messages per IMAP FETCH (default 50; lower if a server rejects large batches) |
| `imap_timeout` | seconds one IMAP *read* may take (default 90). Per read, not per fetch: a big batch over a slow link keeps resetting it and is never cut off. What it does end is a Gmail that has gone quiet in the middle of a command — without it that read simply never returns, and the thread waiting on it never comes back (`0`: no limit) |
| `prune_deleted` | default `true`: after an `index`, drop what the configured folders no longer list (see *When you delete mail*). Read-only either way — this never touches Gmail |
| `prune_max_fraction` | refuse the removal when more than this share of a folder is missing (default `0.25`); override with `--prune-force` |
| `prune_restore` | default `true`: re-index removed mail that turned up in the folder again |
| `reconcile_batch_size` | UIDs per header request while matching folder to index (default 500, with a 1 s pause between) |
| `restore_probe_max` | how many not-yet-indexed UIDs of a folder may be examined for a restore per run (default 2000) |
| `dim` | 768 (full) or MRL 512/256/128 (queries and docs must match) |
| `query_prompt` / `doc_prompt` | task prompts; set `""` if the endpoint handles prompts server-side |
| `thread_context_depth` / `thread_context_chars` | how many prior mails / how many chars of each to include |
| `embed_audio` | embed raw audio attachments (endpoint must support it) |
| `pdf_pages_as_images` | render visual-only PDFs to page images (needs `PyMuPDF`) |
| `embed_media_multimodal` | try the multimodal object form first (default `true`); set `false` for text-only endpoints to skip the probe request |
| `media_item_shape` | how a media item is wrapped for an OpenAI-style endpoint: `"image_url"` (flat part object — OpenAI, vLLM, TEI, `embed_server.py`) or `"content"` (chat-style parts nested per input item). `test_embed_media.py` prints which one your endpoint accepts |

Chunking (multi-resolution):

| key | meaning |
|---|---|
| `chunk_sizes` | window size in characters per tier (default `{"small": 320, "large": 1200, "xlarge": 6000}`) |
| `chunk_overlap_frac` | overlap between windows of a tier, as a fraction of its size (default `0.15`; `0` disables) |
| `max_source_chars` | whole-message budget per source text (default 6000); `xlarge` deliberately equals it |
| `max_windows_per_source` | hard cap on windows per tier per source (default 24). Windows end on block boundaries, so they hold *less* than the nominal size and a full `max_source_chars` source can need more than the naive `ceil(6000 / (320*0.85)) + 1 = 24` — measure your own mailbox with `cap_audit.py` (a 4 452-message one needed 28) and set the cap above its worst case |
| `index_tiers` | derived tiers to build at index time; the canonical whole-source chunk is **always** written |
| `tier_weights` | optional per-tier multiplier on the dense leg, e.g. `{"small": 1.0, "xlarge": 0.9}` |

Retrieval (hybrid):

| key | meaning |
|---|---|
| `search_mode` | `hybrid` (default), `semantic` (vectors only) or `lexical` (BM25 only, works with no endpoint at all) |
| `fts_tokenizer` | FTS5 tokenizer string (default `porter unicode61 remove_diacritics 2`) — changing it needs `reindex --rebuild-fts` |
| `fts_meta_weight` | BM25 weight of the metadata column vs the text column (default 1.5) |
| `rrf_k` | Reciprocal Rank Fusion damping constant (60 is the standard) |
| `agg_k` | a message scores as the **mean of its top-k chunks** (default 3) |
| `lexical_pool` | chunks pulled from BM25 before fusion (default 200) |
| `dense_pool` | `0` scores every chunk by brute force (fine up to ~100k chunks); set a number to pre-cut |
| `w_dense` / `w_lex` | RRF weight of each leg (default 0.5 / 0.5) |

Previews of pictures (the GUI's thumbnails — see *The mailbox GUI*):

| key | meaning |
|---|---|
| `store_thumbs` | default `true`: cut a preview of every image attachment at index time and keep it next to the index. `false` writes nothing and the GUI goes back to fetching every picture from Gmail |
| `thumb_max_side` | longest side of a preview in px (default 320). A 4 000-message mailbox of previews lands around 40–60 MB at this size |
| `media_cache_mb` | budget for the preview file in MB (default 512); over it the least-recently-seen previews go, `media --backfill` cuts them again |
| `media_db_path` | where previews live; `""` means `<db_path>.media.db`, beside the index. Deliberately **not** a table in `mailbox.db`: that file is warm vectors, and a preview should never make a search start-up heavier, nor a `--db` baseline copy dirtier |

Values can also come from env vars: `GMAIL_USER`, `GMAIL_APP_PASSWORD`,
`EMBED_ENDPOINT`, `EMBED_API_KEY`, `EMBED_MODEL` (env wins over the file).

## Usage

```bash
# First run: index everything in the configured folders
python mailbox.py index --full

# Or a small test run first (50 most recent messages)
python mailbox.py index --limit 50

# Later runs are incremental (only mail newer than the last indexed date)
python mailbox.py index

# Indexing is newest-first and resumable: if you Ctrl-C a --full run mid-way,
# just re-run it - it picks up from where it left off (a watermark) and skips
# what's already indexed. Recent mail is searchable as soon as its batch is done.
python mailbox.py index --full   # (re-run after an interruption)

# If a run fails with an IMAP error, re-run with --debug to see every byte
python mailbox.py index --limit 50 --debug

# Search (hybrid by default: vectors + BM25, fused)
python mailbox.py search "the invoice for the Flexa project"
python mailbox.py search "8812-KX" --explain      # per-leg ranks + the RRF maths
python mailbox.py search "factuur van flexa sinds maart"
python mailbox.py search "factuur" --mode lexical  # BM25 only: needs no endpoint
python mailbox.py search "dashboard" --tiers small,large
python mailbox.py search "invoice" --json
python mailbox.py search "transfer code" --log eval/queries.jsonl

# Ranking knobs worth knowing (all of them live in config.json too):
#   --min-relevance 0.30   say "nothing found" instead of showing junk
#   --half-life 240 --recency-weight 0.35   bounded recency prior
#   --demote-transactional 0.55              newsletters, not your mail
#   --fusion-pool 200      how many messages each leg may hand to RRF
#   --no-collapse / --no-filters / --no-tier-norm   turn a fix off to see it
python mailbox.py search "newsletter" --demote-transactional 1 --no-collapse

# Interactive search: loads the index once, then answers from memory
python mailbox.py shell
#   mbox> factuur flexa            a query is typed plain
#   mbox> :top 20  :min_relevance 0.4  :explain on  :json on  :mode lexical

# The same search as a mailbox-style UI, in the browser on 127.0.0.1
python mailbox.py gui                 # or: python gui.py --port 8765 --no-browser

# Compare two settings against the labelled eval set, with a paired bootstrap
python mailbox.py bench --eval-file eval/eval.jsonl \
    --variant new= --variant old=profile=legacy
python mailbox.py evalgen --n 60 --out eval/eval.proposed.jsonl
python mailbox.py label "factuur flexa" 1        # mark result 1 relevant
python mailbox.py audit                          # duplicates, quotes, FTS probes

# Rebuild tiers / BM25 / text from what is already in the database (no IMAP)
python mailbox.py reindex --dry-run         # prints what it costs, writes nothing
python mailbox.py reindex --clean           # text hygiene: quoted replies,
                                            # footers, =?UTF-8?Q?= subjects
python mailbox.py reindex --rebuild-fts     # BM25 only: costs zero embeddings
python mailbox.py reindex --tiers small     # adds the small tier: costs embeddings
python mailbox.py reindex --re-embed        # new prompt or model: re-embeds text

# Index statistics (per-tier counts, fts docs, vector bytes, and what Gmail
# no longer has)
python mailbox.py stats

# Previews: the small pictures the GUI shows without asking Gmail (see
# *The mailbox GUI*). Reads Gmail only for --backfill and --links, and only
# with read verbs; every flag works with --dry-run first.
python mailbox.py media                     # what is kept, and which mail has none yet
python mailbox.py media --backfill          # cut the missing ones, newest mail first
python mailbox.py media --backfill --limit 500
python mailbox.py media --links             # the Gmail link for mail indexed before it was stored
python mailbox.py media --prune-orphans     # previews of mail that is no longer indexed
python mailbox.py media --clear             # delete every preview, keep the index

# Drop what the configured folders no longer list (mail you deleted or archived
# in Gmail). Read-only against Gmail; --dry-run writes nothing at all.
python mailbox.py prune --dry-run
python mailbox.py prune
python mailbox.py index --no-prune    # a run without the reconcile

# Any command can point at another database - e.g. a frozen copy to measure
python mailbox.py --db baseline-legacy.db --profile legacy stats
```

A `reindex` run leaves behind nothing that it created: it replaces the windows of
the tiers it builds, deletes the windows a source no longer derives (a repaired
body that got shorter keeps no trace of the longer one), and prunes the vectors
whose chunk row is gone. Vectors of removed chunks cost disk and copy time but
never recall, since the index is loaded through `chunks`; the file itself only
shrinks when you `VACUUM` it (`sqlite3 mailbox.db "VACUUM"`).

### Profiles: `v2` and `legacy`

`--profile legacy` swaps the tuned ranking knobs for the pre-tuning pipeline
(AND-only BM25 without variants, no filters, no tier standardisation, no
collapse, no priors, the old `Document:` / `SearchQuery:` prompt shape). It is
there for two things: measuring the new ranking against the old one in one run
(`bench --variant old=profile=legacy`), and rolling back in one word if the new
behaviour ever gets in the way.

Naming a profile applies that profile's whole set of knobs, over `config.json`,
in both directions: a file that lists every knob must not be able to turn the
switch into a no-op, and `--profile v2` has to undo a `"profile": "legacy"` left
in the file. Keys that this overwrites are listed in a warning, and a value set
as a shell variable (`EMBED_MODEL` and friends) outranks a profile. If you want
a value to survive a profile switch, pass it as a flag.

### Upgrading an existing `mailbox.db`

The first `stats` / `search` / `index` run upgrades the database **in place**:
it adds the tier columns and the `chunks_fts` table and fills the BM25 index
from the text already stored. No re-fetch, no re-embedding, so keyword search
works immediately — over the coarse chunks the old index had, which still
misses short quotes buried in long bodies. For the finer tiers:

```bash
python mailbox.py reindex --dry-run    # prints exactly how many embeddings it costs
python mailbox.py reindex              # backfills the missing tiers
```

Only the *derived* windows get embedded: the canonical chunk of every source
already holds the complete indexed text, and `reindex` re-derives the tiers
from it — that is why this needs no IMAP. On a real 2 501-message mailbox
(6 417 chunks, 2 759 sources) that backfill measures **29 986 new chunks ≈ 938
endpoint calls** at `batch_size 32`; `--dry-run` prints the number for *your*
mailbox before anything is written, and `--tiers small` alone is the cheaper
first step (it is where quoted-in-a-long-thread recall comes from). It runs
offline from the DB, so it can be interrupted and re-run, and it is idempotent:
a second run rebuilds the same windows instead of appending them.

### Recovering from the window-cap warning

```
note: 182 source(s) exceeded max_windows_per_source, dropping 252 window(s); ...
```

Not an error: the cap only trims the *derived* windows of a tier, the canonical
whole-source chunk still holds the complete text and BM25 still indexes it, so
the only loss is dense coverage of the tail of those long sources. Raising the
cap fixes future runs, but **re-running `index` does not repair what is already
stored** — `index` skips message-ids it already has. Re-derive offline instead:

```bash
python cap_audit.py                 # read-only: worst case per tier, what is
                                    # missing, and what each reindex costs
python mailbox.py reindex --dry-run
python mailbox.py reindex --tiers small
```

`reindex` rebuilds *every* derived window of the tiers it builds, so a few
hundred missing tails can cost a full re-embedding of the tier (45 503 small
chunks ≈ 1 422 calls on a 4 452-message mailbox). `reindex_tails.py` compares
what the current config derives against what the database holds and touches
only the messages that are short — same 238 missing windows for ~4 700 chunks
≈ 150 calls:

```bash
python reindex_tails.py --dry-run
python reindex_tails.py             # small tier by default; idempotent
```

### Switching embedding models

Vectors from two different models are not comparable even at the same
dimension — the build and the quantisation move the space — so a mixed index
does not fail, it just scores worse, and nothing in a cosine search looks wrong.
`index` therefore records the model, dim and server it embedded with, and the
next run says so:

```
WARNING vectors were embedded with embed_model='embeddinggemma-2:latest',
config says 'embeddinggemma-2-Q8_0' - old and new vectors are different spaces,
so scores mix; run: python mailbox.py reindex --re-embed
```

(A changed server name warns too: the same model name on another host is not
necessarily the same build.) `reindex --re-embed` rebuilds every *text* vector
from the text already in the database — no IMAP. Media vectors are the one
exception: image and audio bytes are never stored in the index, only their
vectors, so those chunks keep the old ones and `reindex` prints how many are left
over; `index --full` re-fetches and re-embeds them. (A picture's preview sits in
the sidecar next to the index, but it is a 320 px re-encode for the GUI, not the
bytes a media vector was made from.) On the 2 501-message mailbox that
split is 5 281 text chunks re-embedded offline, 1 136 media chunks needing a
re-fetch.

### How retrieval works

1. **Three resolutions.** Every source (body, metadata block, text attachment)
   is chunked at `small` / `large` / `xlarge`. The tier whose window size fits
   the whole capped text is the *canonical* chunk: one row holding the complete
   text, labelled with the bare source (`body`). Smaller tiers contribute their
   windows, labelled `body [small 3/9]`; a window identical to one already
   stored (repeated footers, quoted boilerplate) is dropped instead of
   embedded twice.
2. **Two legs.** The dense leg embeds the query and scores every chunk by
   cosine similarity. The lexical leg runs FTS5 BM25 over three columns:
   `meta` (`From/To/Cc/Subject/Date/Attachments`, weighted `fts_meta_weight`
   above the body), `text`, and `quoted` — the dragged-along half of a body,
   still findable at `fts_quoted_weight` without its text in the vectors.
   Query-side the text is cleaned first: stopwords are dropped (the index has
   no stoplist, so a bare `de` would be a *required* term), every word is
   joined to its own forms with OR (`facturen` also matches `factuur`), terms
   are combined with AND plus a `*` prefix on long terms, and a query whose AND
   matches nothing is retried as OR — a word typed slightly differently should
   cost relevance, not return zero results. There is deliberately **no porter
   stemmer**: it is English-only, and it pulled Dutch forms apart so `domein`
   and `domeinen` stopped meeting each other.
3. **Filters as SQL, before the search.** `mail van flexa sinds maart` is
   two questions. `queryparse.py` reads the sender, recipient, subject, date
   range and "met bijlage" out of the sentence and they become a **SQL
   pre-filter**; only what is left gets embedded. Putting `van flexa` in the
   embedded text moves the vector toward the word *van* instead of restricting
   the candidates. `--no-filters` turns it off; the printed `Filters:` line and
   `--explain` show what was taken out.
   Two guards keep a guess from destroying the answer. A bare `van` only becomes
   a sender when what follows could be one: `van je order` and `up to date` name
   nobody, and a wrong sender filter does not lose a point of precision, it
   throws the result set away. And a parsed filter that matches **no message at
   all** is dropped, so the query is searched as typed - `Afgeleverd: je pakket
   van Lyon` has no sender called Lyon. A filter you typed yourself
   (`from:nobody@x.nl`) stays decisive, because there the emptiness is the
   answer.
4. **Fusion.** Each leg's message list is cut to `--fusion-pool` before
   anything is added: `w/(rrf_k + rank)` from beyond the other leg's reach is a
   different quantity, and uncapped lists mix two rank scales. `rrf_k` is 20 —
   60 flattens the differences between ranks 1, 2 and 3, which is exactly where
   the decisions are — and the sum is divided by its analytic maximum, so
   `fused` means the same thing for every query (1.0 = rank 1 in both legs).
   That comparability across queries is what makes a score floor possible.
5. **Per message.** Chunks fold up to their message as the **mean of the top
   `agg_k` chunks** — a plain max lets one long message win on volume, a plain
   mean buries a single exact hit — after dropping windows with an identical
   `text_hash` and windows of the same message that are ≥ `--window-dedupe`
   cosine apart. Cosines are standardised per tier first (`tier_norm`): a
   3 200-character window and a 300-character one do not reach the same cosine
   with the same query, and comparing them raw hands the long chunks the list.
6. **Bounded priors.** Recency (`--half-life`, `--recency-weight`) is
   multiplicative: it can only move a score *down*, by a bounded amount. A prior
   that could also promote is how a search starts preferring yesterday's
   newsletter to the contract you typed four words about. The transactional
   sender prior (`--demote-transactional`) ships **off** (`1.0`) because it was
   measured: at `0.55` it cost 0.037 recall@5 [0.010, 0.072] and 0.045 MRR@10
   [0.009, 0.085], and P@5 was *higher* without it, so it was not buying
   precision either. Nearly half of a real inbox is transactional, and the mail
   people search for ("dmarc", "aanmaning", "ssl certificaat vernieuwen") is
   sent by those same addresses: a flat penalty punishes the answer, not the
   noise. It pushed one correct answer from rank 3 to rank 19.
7. **Duplicate results are collapsed.** Results collapse per thread id, per
   identical body fingerprint, and above `--collapse-sim` cosine, so the top of
   the list is ten things you had not already seen; `(+N in thread)` says what
   was folded. The fingerprint comes from the message's whole body, not from the
   chunk that happened to match: a hit is often a window, and two windows of one
   letter have different fingerprints, which is how the same newsletter text
   reached the list three times. Without this the answer to "transfer code" is
   the same sentence from four mails, and nine other results never appear.
8. **Abstaining.** Below `--min-relevance` the answer is *"no result above the
   relevance floor (best 0.281 < 0.30)"* plus the closest few marked as
   reference, instead of ten plausible-looking wrong mails. An empty-looking
   inbox and a wrong one are different problems, but on a measured inbox the
   score cannot tell them apart (see "What the last measurement said"), so the
   default floor is deliberately low and this mostly fires on filters that match
   nothing. It is a guard, not a feature, until the score carries the signal.
9. **Optional rerank.** `--rerank endpoint` cross-encodes the top
   `--rerank-candidates` messages through a Jina-compatible `/v1/rerank`
   (llama.cpp: `llama-server --rerank`). It stays off by default, and not from
   taste: the gate is a measured gain whose interval excludes zero, and this eval
   (70 scored queries, MRR@10 resolved to about +-0.10) can show none either way.
   The cases it is for are visible: "ssl certificaat vernieuwen" ranks the right
   mail 11th on cosine and 31st on BM25, which no amount of weight tuning fixes.
   The lever is real and the instrument to price it is not, so the labelled set
   grows before the reranker gets a default.

### The `--time-weight` toggle

The old additive blend, kept because it is blunt and predictable: the score
becomes `(1-alpha) * fused + alpha * exp(-days/30)`, with `--alpha 0.3` by
default. It **replaces** the bounded recency prior instead of stacking on it —
two recency terms in one score is how a ranking starts ignoring the words you
typed. Use it when you want recency to *be* the ranking, not to nudge it.

## When you delete mail

Deleting mail happens in Gmail. **This app cannot delete anything, and never
will**: the IMAP layer only issues read verbs (`LOGIN`, `LIST`, `NAMESPACE`,
`EXAMINE`, `UID SEARCH`, `UID FETCH … BODY.PEEK[…]`, `NOOP`, `CHECK`, `LOGOUT`)
— there is no `STORE`, `COPY`, `APPEND`, `MOVE` or `EXPUNGE` anywhere in the
code, and no button that would ask for one. `test_mailbox` pins that down: the
fake connection the reconcile tests run against raises on any write verb *and*
on a read-write `SELECT`, and `GmailFetcher.READ_ONLY_COMMANDS` is the list it
checks.

What the index does instead is keep its half of the bargain. The configured
`folders` are **mirrored**: after every `index` run — and so after every
**Refresh** in the GUI — each folder is listed once, and every stored row of
that folder the folder no longer lists leaves the index: message row, chunks,
vectors, FTS entries and the previews cut from its pictures together — no orphan
row, no orphan thumbnail. Archiving out of `INBOX` counts
the same as deleting: the mail is not in the folder, so it is not in the index.

Mail that comes back is picked up again. A normal `index` only fetches mail
newer than its watermark, so an un-deleted message would otherwise be invisible
forever; every removal is therefore recorded in `removed_messages` with its UID,
and a reconcile that finds such a UID in the folder again re-fetches and
re-indexes it and clears the entry.

Every check refuses rather than half-does the job:

| situation | what the run does |
|---|---|
| a folder cannot be listed or `EXAMINE` fails | removes nothing there, says it was skipped |
| a folder lists **0** messages while rows are stored | not believed — skipped |
| `UIDVALIDITY` changed | the folder was re-created under a new numbering: stored UIDs are re-mapped, nothing removed |
| more than `prune_max_fraction` of a folder is missing | skipped, until you say `--prune-force` |
| `index --limit`, or `--db` pointing somewhere else | the reconcile is skipped: neither describes the whole mailbox in the configured database |
| login or connection error | nothing is removed; the run says the check did not happen |

```bash
python mailbox.py prune --dry-run   # what would go, listed, and nothing written
python mailbox.py prune             # the same, applied
python mailbox.py prune --verbose   # also list them after a real run
python mailbox.py index --no-prune  # index without reconciling
python mailbox.py stats             # removed total, last reconcile, newest 10
```

The cost is one `UID SEARCH ALL` per folder once UIDs are recorded — about a
second for a 4 000-message INBOX. The **first** run against an index written
before UIDs were stored reads one header per message as well
(`UID FETCH … BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)]`: ~8 s and ~340 KB for
4 457 messages) to match folder rows to database rows; the UIDs it learns are
stored, so that is once per folder, and a `--dry-run` does not write them.

A real one, on a 4 457-message index of an account where 346 messages had been
deleted:

```
reconciled INBOX: 4158 live, 4457 stored, 346 removed, 4111 uids recorded, headers swept
store now: 4111 messages, 45072 chunks, 45072 fts docs.
```

The run after it is one line and about a second — `reconciled INBOX: 4158 live,
4111 stored` — and that line is why it can ride along with every refresh.

The mirror is reported in both directions, because the other one is worth
knowing too. A UID the folder lists with no row in this index was **never
fetched** — mail an old run skipped, or mail an incremental run's watermark
walked past — and this check cannot fetch it (it is a check, not an index), so
it says so: `47 in the folder were never indexed`, with the hint that
`index --full` is what picks it up. A full run re-downloads the mailbox but
skips every message id it already has, so it re-embeds only what was actually
missing. On this mailbox that number is 47 real messages from August 2021, and
until someone runs `--full` the honest state of the index is *47 short*, which
is a better thing for the tool to say than to be quietly smaller.

In the GUI the same facts are where you would look for them: the status dot's
tooltip carries `346 messages Gmail no longer has, last dropped 21 Oct 16:52`
(the removal's own timestamp, in the page's clock, not the time the folders were
last checked), a finished Refresh says how many this run dropped (or re-indexed
because they came back), and opening one of those messages — from an old link or
a bookmark — answers

```
removed from the index on 2026-10-21: the folder INBOX no longer holds this
message, so the last refresh dropped it. Nothing in this app deletes mail —
look in Gmail.
```

Two consequences of removing rows for real: the file does not shrink by itself
(`sqlite3 mailbox.db "VACUUM"` gives the space back), and `eval/eval.jsonl`
names message ids, so a prune moves `bench` and `eval` — a labelled message that
is gone from Gmail is gone from the metrics too, which is a true result rather
than a regression.

## The mailbox GUI

```bash
python mailbox.py gui              # opens http://127.0.0.1:8765/ in your browser
python gui.py --port 9000 --no-browser --db other.db
```

It needs `fastapi` and `uvicorn` (both in `requirements.txt`) and nothing else:
the front end is three files in `web/`, served as static files, with no build
step, no bundler and no CDN.

Three panes, in the shape your mail already has: a rail (view, totals, senders
by volume), a result list, and a reading pane. It is the CLI's search, not a
simplified one: `hybrid` / `semantic` / `lexical`, and every ranking knob in
`config.json` in the tuning panel, so a result here and a result from
`mailbox.py search` are the same arithmetic.

What it shows that a terminal cannot:

- **Both legs of every result.** Each row carries two hairlines — where the
  vector leg ranked it, where BM25 did. A keyword match with no semantic
  backing and a semantic match that ignores your exact word look identical in a
  single score and are completely different things to act on.
- **Chips, not a rewritten query.** `factuur van flexa sinds maart` is one
  sentence to you and three constraints to the engine, so the parsed filters
  come back as removable chips. Dropping one re-searches the *whole* sentence
  with the parser off for that request: re-reading it would cut `van flexa` out
  of the embedded text again and resurrect the filter you just removed.
- **Why an answer is not an answer.** Results under the relevance floor are
  labelled as the closest mail rather than an answer; a filter chip that matches
  nothing says which chip emptied the list; if the embedding endpoint is down
  the banner says these results are keyword-only rather than pretending the
  vector leg ran.
- **The text that made it a hit.** The reading pane shows the whole stored body,
  the matched window if it is not the whole thing, a link to Gmail where a body
  cut at `max_source_chars` continues, and what was quoted. The pane names what
  the reader can do next; the config key behind a cut stays in this file.

**Searching *with* an image.** The paperclip in the search bar takes a picture —
or paste a screenshot, or drop a file on the bar — and it sits next to the words
as a thumbnail you can remove again. Words and picture are not averaged: each
chunk is scored by whichever side it matches better, and the words still run
through BM25. So *de foto van het onderdeel, uit de bestelmail* can be carried by
the picture, by the words, or by a mail that matches one side well and the other
not at all — averaging would sink all three. A picture on its own leaves no
keywords to fuse, so the vector leg answers alone and the banner says so rather
than showing a bm25 rank that was never computed. PNG, JPEG, WebP or GIF; SVG is
refused, because an SVG is a script that draws. A 24 MB phone photo is not
refused either — the browser redraws it at the 1024 px the model would have
received anyway, showing *reducing 24.0 MB photo…* while it does, and the
thumbnail's tooltip then says what was really sent (`image001.jpg · 184 KB sent ·
reduced from 24.0 MB to 1024×768 px, which is the side the model sees`). A file
already under the 12 MB cap goes out untouched: a file byte-identical to an
attachment lands on the same vector, and that is the one case where the search is
exact.

Two conditions, and the GUI says which one you are missing. The endpoint has to
read pixels (*Checking whether images are really embedded* below), and the
attachment has to have been indexed from pixels — an image stored as a filename
vector cannot be met by a query image, no more than a word can be met by a
photograph. The status dot carries both answers, and a search with a picture
refuses loudly (`409`) instead of quietly answering with the keyword leg: BM25
cannot see an image, so there is nothing to degrade to. The bytes travel
browser → endpoint only; nothing is stored, and no response echoes them back,
which is why the thumbnail is drawn from the file you picked, in the tab.

Browsing is separate from searching on purpose: the inbox view pages by keyset
(`date < cursor`, so mail arriving mid-scroll cannot make you skip or repeat a
message) and its text box is a `LIKE` over subject, sender and preview — not
BM25. Typing a sender's name in a list must not start surfacing 2019.

**Refresh** runs `mailbox.py index` as a subprocess and tails its progress in a
panel; the index is re-opened when it finishes, so the warm snapshot never
outlives the data. That run also reconciles the folders, which is what makes a
Refresh in Gmail show up here (see *When you delete mail*) — so a refresh can
take messages *out*, and the toast that finishes says how many. Output goes to
`gui-index.log`.

Three things the GUI cannot do, and one it can. It cannot mark mail read (the
`\Seen` flag is not in this schema). It cannot delete, archive or otherwise
change a message: the button that removes something from this index is the one
in Gmail, and the index follows it. `Read/unread`, download and delete are
IMAP's, not this index's — the app has no IMAP write verb to offer, and adding
one is not on the roadmap. But it **can open attachments**, and a picture opens
without Gmail being asked at all: a preview of every image attachment is cut at
index time — from bytes the indexer decoded anyway for its embedding — and kept
beside the index (`store_thumbs`). The thumbnail in a list row, the one on an
attachment chip and the first frame of the preview box all come off that file:
`GET …/media/<name>` reads it off disk, `ETag` plus `immutable` make the second
look cost nothing, and the IMAP connection is not opened. A preview is a re-encode
at `thumb_max_side` px and never the original — JPEG, unless the picture has pixels
that really are see-through, which is the one case PNG's bytes are worth; an alpha
channel that is opaque everywhere is not that case, and asking the pixels rather
than the colour mode is what kept a mailbox's largest previews from being 190 KB
copies of a 320 px picture. The file itself stays one click away in that same box,
and a file with no preview answers with the command that cuts one
(`python mailbox.py media --backfill`).

Everything that is not a preview comes from the account when it is opened: the
click re-reads that message from the folder and UID its row recorded, and it reads
**one MIME part rather than the mail**. Which part, in three steps, cheapest first:
a section already learned for that file (table `parts` in `mailbox.media.db`);
otherwise `BODYSTRUCTURE`, which names every part of the message, its filename and
its encoded size, in a single reply — and every section that reply names is filed,
so the next attachment of the same mail is one command away instead of another
search; otherwise a `BODY.PEEK[k.MIME]` probe, asked in short runs. Whatever a step
suggests, the bytes are served only after that section is re-read and *its own*
headers name the same file, so a wrong guess costs a slower click and never a
different document. All of it inside `READ_ONLY_COMMANDS`: opening a PDF cannot
even mark the mail read.

The probe is deliberately narrow (`PROBE_CHUNK`) and stops at the first section that
does not answer, because Gmail fails an entire `UID FETCH` if one named section is
missing — `NO … Some messages could not be FETCHed (Failure)`, and whatever the
other sections in that command would have said is lost with it. A blind probe of
sixteen sections on a three-part mail therefore looks exactly like a probe that
found nothing, which is how this path stopped working in production while the suite
stayed green: the fake server answered the sections a message had and kept silent
about the ones it had not, where the real server refuses the whole question.
`FakeIMAP` refuses them the way Gmail does, and answers nothing for
`BODYSTRUCTURE` when a test asks it to (`no_structure=True`), because that is the
only way the probe path is still covered. Reading the whole message stays the last
resort, and the only route to a file the parts do not carry; the background
`media --backfill` job never takes it, since a 320 px preview is not worth 5 MB.

Images, PDFs, audio/video and plain text open in the tab; anything that could
execute from this origin (HTML, SVG, scripts) is download-only, never inline —
for *files*. An HTML **body** is the one thing that does get rendered, and it is
never served as a document: see *Reading a mail as mail*. The
bytes reach your browser's cache and no disk here, so that click needs Gmail
reachable; a file or message that is gone at the source answers with the date it
left the index, not a stack trace. A row indexed before UIDs were recorded is
traced by its `Message-ID` header instead.

Previews are the one place this app stores something of the mail itself beyond
text, which is why they are the one thing you can take back: `store_thumbs: false`
stops new ones, `media --prune-orphans` drops the ones whose message is gone, and
`media --clear` deletes every one of them (the index and its search keep working,
the GUI just goes back to asking Gmail for each picture).

What it does say, since mail can leave: the status dot carries `N messages Gmail
no longer has, last dropped on …`, a finished Refresh names the removals it just
made, and a message that is gone answers with the date it was dropped and which
folder stopped holding it instead of `no such message`.

### Reading a mail as mail

An HTML mail has two bodies and this index searched one of them. The text part is
what was chunked, embedded and highlighted; the HTML part is what the sender meant
you to see, and it is often a different document — a receipt that says nothing in
text but shows a table of amounts. So the reading pane's body row carries a switch,
`as text` and `as mail`, in the same segmented control the search bar uses for its
ranking modes, and the header renames itself to `html part` when you flip it. Under
the frame a strip says which part is on screen, which charset it was read as,
whether Gmail was asked at all, and *not the part the index searched*: a pane that
silently swaps which of two parts it is showing is a pane you stop trusting when
the two disagree.

Nothing from the mail is ever written into this page. `GET /api/message/<id>/html`
answers **JSON**, never `text/html` — HTML with this app's origin would be this
app, with a stranger's markup made first-party in it — and the page assigns the
returned document to an iframe's `srcdoc` with
`sandbox="allow-popups allow-popups-to-escape-sandbox"`. No `allow-scripts`, so the
`<script>` a mail carries is inert text; no `allow-same-origin`, so the frame has no
origin at all: it cannot call `/api/*`, cannot read the page around it, cannot
follow a redirect back to the app as itself. `srcdoc` is *assigned*, so there is no
moment where mail markup is this document's subtree — which is also why the two
front-end scripts are scanned for the four words that would create one. That is
Gmail's shape, not merely Gmail's escape rules: the mail is rendered *somewhere
else*. A link is the one thing allowed to leave (`allow-popups`, and the popup
escapes the sandbox so it is the target's page rather than a frame that could script
its opener), because a link in a mail is meant to be followed by going there.

`mailhtml.py` is the only file that touches the sanitizer — `nh3`, the Rust
`ammonia` library, which parses with html5ever: the same parser the browser uses, so
the tree this app sanitizes is the tree the frame will render rather than a similar
one. The profile is email rather than "blog comment": `table`, `td`, `font`,
`center`, `align`, `valign`, `bgcolor`, `cellspacing` and a head `<style>` all
survive, because a newsletter that renders as an apology is not safer; `display:none`
survives, because a mail that hides its plain-text copy is normal. Stripped:
`position`/`z-index`/transforms/animations (a mail may not cover the page it is
in), `@font-face` and `@import` (nothing fetched, nothing downloaded), forms and
fake login boxes, `iframe`/`object`/`embed`, `<base>` (it would redirect every other
URL in the document), `meta` refresh, and comments. Stylesheets are re-emitted from
parsed CSS rather than copied, declaration by declaration. Every URL must be
`http(s)`, `mailto:`, `tel:` or `cid:` — and *relative* URLs are refused too, because
a frame with no document URL inherits the page's base URL, which makes a mail's
`src="/api/status"` a request to this app. `srcset` is not whitelisted at all: `nh3`
does not run its scheme allowlist over that attribute, and an image source that
cannot be checked is not worth a responsive logo.

Images are the one part that talks back, so they stay off until asked. A remote
`<img>` becomes a 1×1 transparent pixel carrying the sender's own width and height,
which keeps the layout Gmail would have shown and sends nothing; `display images`
re-renders with `img-src http: https: data:` written into the frame's policy *before*
the markup. `cid:` references resolve to that message's own attachment URL, from a
`Content-ID` map read once off the part headers. Deliberately **not** built: an image
proxy. Gmail re-hosts images through its own servers so your address never reaches the
sender; here, clicking display images means your browser contacts the servers the
mail names, with `referrerpolicy="no-referrer"` so it does not learn which mailbox
was read. That is the one thing Gmail does for you that this app does not, and it is
a visible trade rather than a silent one.

Getting the part costs one narrow read, and after that nothing: the section is
remembered in the `parts` table under the reserved name `#text/html` the way an
attachment's section is, the raw HTML is then cached in `mailbox.media.db` beside the
previews (`#text/html`, plus `#text/html+cid` for the id map), inside the same
`media_cache_mb` budget and the same LRU, and it is deleted with its message. It is a
cache and not a store: nothing is re-indexed, `--profile legacy` is untouched, and a
second look at the same mail issues no IMAP command at all. Fetching it is still
`EXAMINE` plus `BODY.PEEK[…]`, so reading a mail here does not mark it read in
Gmail. `render_html` turns the view off; `render_html_max_bytes` (2 MB) and the
400-level nesting limit are checked *before* parsing, because 100 000 nested `<div>`s
measured 21.7 s in the sanitizer on a worker whose job is to answer a click in a
second — a refusal in a sentence, with the text still on screen. `show_remote_images`
is the opposite default of the same name in a mail client: nobody should open a
newsletter for the first time and immediately be counted by forty servers.

Two things are honest omissions rather than gaps. The frame has a fixed height with
a drag handle (`resize: vertical`) instead of measuring the mail: a frame with no
origin cannot report its own height to the page, and the permission that would fix
that is the one this view exists to withhold. And the render is not the search: a
phrase that only exists in the HTML of a mail will not be found by this index, which
searches text, and the strip under the frame says so on every mail it renders.

**Bind to `127.0.0.1` and leave it there.** There is no authentication, and your
whole mailbox is behind it. `--host 0.0.0.0` puts an unauthenticated mailbox on
your network; `--host 127.0.0.1` is also the default of the raw `gui.py`. The
previews are part of that same boundary: `mailbox.media.db` holds pictures of your
mail — and, since the mail view, the HTML body of every mail you opened — in plain
view of anything that can read the file, which on your own machine is
the same set of readers that can already read `mailbox.db` — and on a machine that
is not yours is a reason for `store_thumbs: false`.

Under the hood, all database work runs on **one worker thread**. That keeps the
SQLite connection on the thread that created it — no `check_same_thread`
fudging — and serialises searches, which is what you want when a search is
already reading 146 MB of vectors: two concurrent queries make both slower, they
do not make one faster. Files are the exception, and they run on **two separate
workers**: reading one back from Gmail takes a second or two, and a search that
queues behind a PDF is the difference between a tool that feels broken and one
that does not. The status line counts the searches that are waiting; opening a
cold file never adds to it. Previews, which come off disk, are served by the same
two workers and never touch the search queue at all.

## What gets indexed per message

- **body** — plain text (HTML converted), capped at `max_source_chars` and
  indexed at all three tiers (see *How retrieval works*)
- **meta** — `From / To / Cc / Date / Subject / Attachments: <extensions>`
  plus a **Thread** block: the N most recent prior mails in the same thread
  (grouped by the `References` / `In-Reply-To` chain — Gmail's `X-GM-THRID` is
  what the code asks for first, but it is an IMAP fetch item rather than a
  header in the body, so over IMAP the chain is what actually groups),
  so a message is retrievable by its conversation
- **attachments** — images embedded directly (multimodal; if the endpoint
  only accepts strings, the client falls back to the attachment's text
  label — see the endpoint contract below), PDF/DOCX/XLSX/PPTX
  text-extracted, audio optionally embedded, other types metadata-only.
  A media chunk is one indivisible item, so it is its own canonical row at
  tier `media`: never windowed, never re-derived by `reindex`, and reachable
  through the dense leg only (there is no text to BM25).
- **gmail_url** — a direct web link to the message, built from Gmail's message
  id (decimal -> hex): `https://mail.google.com/mail/u/0/#all/<hex>`. Stored per
  message and shown in `search` output (a `gmail:` line) for any message that has
  one. **It is empty for mail fetched as raw bytes**: the parser reads
  `X-GM-MSGID` from the fetched RFC822 body, and Gmail does not put its own
  `X-GM-*` headers *in* that body — they are IMAP *fetch items* (`UID FETCH n
  X-GM-MSGID`), a different request than the `(UID RFC822)` this indexer asks
  for. So a mailbox indexed before that was known has 0 URLs stored: not a broken
  derivation but a value that was never in the bytes. `media --links` asks for the
  fetch item for exactly the rows still missing one, in UID batches, and writes
  them — no re-index, no re-embedding, and the account is only ever asked to
  `EXAMINE` and `FETCH`. A row with no UID cannot be asked about, and is reported
  as such. `X-GM-THRID`, which the thread grouping
  also prefers, is the same kind of item — thread ids here come from the
  `References` / `In-Reply-To` chain, which *is* in the body.
  `https://mail.google.com/mail/u/N/#inbox` still opens the mailbox in the right
  slot (`gmail_account_index`); only the per-message deep link needs the item.

## Embedding endpoint contract

The default adapter in `mailbox.py` (`EmbeddingClient`) speaks an
OpenAI-compatible shape, and also reads Ollama's native reply so either URL
works:

```
POST {embed_endpoint}
Authorization: Bearer {embed_api_key}
{"input": ["title: Factuur 8812-KX | text: Beste klant, hierbij ...", ...]}
-> {"data": [{"embedding": [ ... ]}]}        # /v1/embeddings  (OpenAI shape)
-> {"embeddings": [[ ... ], ...]}            # /api/embed      (Ollama native)
```

The strings in `input` are built by `EmbeddingClient.doc_input` /
`query_input`, and the `title:` / `text:` / `task: search result | query:`
shape is part of the model's contract, not decoration: the same text without
it lands in a different part of the space. A `--profile legacy` run sends
`Document: ...` and `SearchQuery: ...` instead, which is why switching profile
means re-embedding (`reindex --re-embed`), not just re-querying.

When `embed_model` is set, the same body also includes `"model": "<embed_model>"`.
Hosts such as Ollama require that field; leave it empty to omit it.

Media items are sent as one object combining the media with its text
label, so a multimodal endpoint returns a single combined embedding:
`{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}, "text": "title: a.png | text: image attachment"}`
(audio uses `type: "audio_url"`). A ready-made multimodal server is
included: `embed_server.py` (wraps EmbeddingGemma 2 via
sentence-transformers, per Google's multimodal guide —
`python embed_server.py --port 1234`). If your endpoint uses a different
request/response format, replace the `EmbeddingClient._post` / `_vectors`
bodies only — the rest of the script is unaffected.

If your endpoint only accepts strings (it answers the object form with
`'input' field must be a string or an array of strings`), the client
detects the rejection on the first media chunk, logs one warning, and
falls back to embedding the media's text label
(`title: <filename> | text: image attachment`) for the rest of the run: indexing
completes, and attachments stay searchable by name/type, just not by
image content. To search by image content, the endpoint must accept
`image_url` items (a multimodal-capable EmbeddingGemma 2 server).

### Checking whether images are *really* embedded

An endpoint can accept image input and silently ignore the bytes — which looks
identical to real image search from `index` output, until you search for
something only the picture shows. `test_embed_media.py` probes it directly
(synthetic screenshots, plus the exact call `index` makes):

```bash
python test_embed_media.py                      # uses config.json
python test_embed_media.py --image path\to\shot.png --query "bar chart"
```

It distinguishes the three cases — images change the vector, images are
accepted but produce a byte-identical vector, images are rejected — and exits
non-zero when attachment content is not actually searchable.

One known case: **Ollama cannot embed images for embedding models.** Its
`/v1/embeddings` rejects `image_url` items outright (`input.image_url must be
a string`), and although native `/api/embed` accepts an `images` array, the
reply is byte-identical with, without and with a different image (verified on
Ollama 0.40.6 with `embeddinggemma-2`, whose `/api/show` even advertises a
`vision` capability). `mailbox.py` detects a native endpoint and indexes
attachments as text with one clear warning, instead of storing filename
vectors under a tier that implies otherwise. For real attachment-content
search, run `embed_server.py` and point `embed_endpoint` at it.

Two endpoints are verified to really embed pixels, and `mailbox.py` negotiates
between their envelopes automatically (the first media chunk tries the
configured shape, then the other, and logs what it settled on):

- **`embed_server.py`** (this repo): flat `{"type": "image_url", ...}` items,
  i.e. `media_item_shape: "image_url"`.
- **a multimodal GGUF under llama.cpp** (e.g. `embeddinggemma-2-Q8_0.gguf`
  started with its `mmproj` file): it rejects the flat item
  (`"prompt" elements must be a string…`) and accepts chat-style parts nested
  per input item, so use `"media_item_shape": "content"`. Measured with a
  "good morning" greeting card (tulips + cappuccino, filename gives nothing
  away): the query *"good morning greeting with purple tulips and a cup of
  coffee"* ranks that attachment first at cos +0.75 against synthetic
  chart/table/sign screenshots (~+0.59), and the same attachment falls to last
  for *"a red stop sign on a white background"* — pixels, not filenames.

An image chunk is only useful once it is in the index: after switching to a
pixel-capable endpoint, the media chunks already stored still hold text-label
vectors from before, so re-embed them (`index --full` re-fetches attachments;
`reindex --re-embed` covers everything else). The same applies to searching
*with* a picture in the GUI: a query image is embedded bare — no filename, no
`title:` prefix, because that is how the document side embeds an attachment and
the two would not meet — so it can only be matched against media chunks that
were embedded from pixels. Text-label vectors are not broken, just deaf to
photographs, and they keep working for filename and type queries.

A query image is never answered with a text fallback. `embed_media` falls back
to the filename vector when an endpoint refuses image input, which is the right
call while indexing a document; applied to a query it would embed the words
"image attachment" and hand back every mail containing them, presented as the
contents of your screenshot. So the GUI's image search fails with the reason
instead, and the status dot then knows this endpoint cannot do it.

## Measuring a change

Search quality is not a feeling. `eval.py` (registered as `bench`, `evalgen`,
`label` and `audit` under `mailbox.py`) scores ranked lists against a labelled
set and, when two settings are compared, resamples the *same* queries so the
question is answered with a confidence interval instead of a decimal:

```bash
python mailbox.py bench --eval-file eval/eval.jsonl --variant new= --variant old=profile=legacy
python mailbox.py bench --eval-file eval/eval.jsonl --baseline baseline.json --offline
```

Every change in this pipeline is justified by a delta whose interval excludes
zero — and a change whose interval includes zero is not a change, whatever the
mean says. What the numbers mean:

- **recall@5 / recall@10** — of the mails you would have wanted, how many are in
  the list you got. This is the one that decides whether the tool is usable.
- **MRR@10** — how far down the first good answer sits (1.0 = first hit).
- **nDCG@10** — the same, but crediting a list that orders the good ones first.
- **P@5** — how much of the top five is junk.
- **abstain** — on queries that *do* have an answer, how often the search said
  "nothing good here" anyway. The rate for the out-of-corpus queries, where
  saying so is the correct answer, is printed with the timings.
- **diversity** — how many separate answers the top 10 actually holds, and how
  many queries had two results with the same body text in it. Recall does not
  notice a list that contains one paragraph four times; a reader notices that
  before it notices anything else.
- **error buckets** — for each miss, the first cause that applies:
  `not_in_corpus` (nothing to find), `missing_from_pool` (never retrieved in the
  top 50 -> a retrieval problem), `buried` (retrieved, ranked too low -> a
  ranking problem), `below_threshold`, `ok`. Fixing ranking will not fix a
  retrieval miss, so the split is the point.

`eval/eval.jsonl` is one JSON object per line: `id`, `query`, `relevant` (the
message ids that answer it), optional `forbidden`, optional `out_of_corpus`.
`evalgen` proposes rows from SQL rules (exact subject, sender, date, attachment
filename) and `eval/curation.py` adds content rows from exact phrases and caps
label lists that are longer than the list a user can see — a rule that matched
38 mails makes recall@10 ≤ 0.26 no matter how good the ranking is, which is a
property of the label, not of the search. **The labels are weak until you look
at them**: delete rows you would never ask, add the answers a rule missed with
`python mailbox.py label "<query>" <rank>`.

The honest source of new queries is real use: `search "…" --log eval/queries.jsonl`
appends every query someone actually typed, with what each leg ranked. Those are
the queries to label next.

### What the last measurement said

84 queries, 14 of them out of the corpus on purpose, 70 scored. Every `current`
row is this database with one setting changed; the baseline row is the frozen
copy of the mailbox from before this work, so it differs in corpus *and* ranking.

```bash
python mailbox.py bench --eval-file eval/eval.jsonl --variant new= \
    --variant demote=demote_transactional=0.55 --variant nocollapse=collapse=false \
    --variant oldranking=profile=legacy --variant lexical=mode=lexical \
    --variant oldprompts=cfg.prompt_scheme=legacy \
    --baseline baseline-legacy-baseline.json --save final.json
```

| run                                       | R@5 | R@10 | P@5 | MRR@10 | nDCG@10 | repeated body |
| ----------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------------- |
| frozen `baseline-legacy.db`               | 0.618 | 0.749 | 0.254 | 0.647 | 0.646 | 20 of 84 |
| current settings                          | 0.767 | 0.790 | 0.334 | 0.805 | 0.763 | 12 of 84 |
| current, `demote_transactional=0.55`      | 0.723 | 0.764 | 0.303 | 0.751 | 0.716 | 14 of 84 |
| current, `collapse=false`                 | 0.808 | 0.859 | 0.371 | 0.793 | 0.802 | 27 of 84 |
| current, `--profile legacy` (old ranking) | 0.623 | 0.691 | 0.240 | 0.632 | 0.635 | 28 of 84 |
| current, `mode=lexical` (BM25 alone)      | 0.661 | 0.679 | 0.280 | 0.621 | 0.607 | 13 of 70 |
| current, old prompt contract              | 0.757 | 0.781 | 0.329 | 0.792 | 0.749 | 12 of 84 |

What the resampling separates at 70 queries, and what it does not:

- **End to end against the frozen baseline: better on four of five metrics.**
  Recall@5 by 0.124 [0.038, 0.215], MRR@10 by 0.131 [0.041, 0.227], nDCG@10 by
  0.097 [0.015, 0.182], P@5 by 0.067 [0.019, 0.121]. Recall@10 (0.034) is the one
  the interval will not resolve, and 70 queries is why: it resolves MRR@10 to
  about +-0.10, so a change has to be worth a tenth to be seen at all. Around 300
  labelled queries would fix that; nothing in the pipeline will.
- **The ranking knobs alone, on the same corpus:** every metric significant
  against `--profile legacy`, MRR@10 by 0.144 [0.062, 0.231]. That variant shares
  the database with the run it is compared against, so it measures ranking, never
  the tokenizer or the cleaned text.
- **A prior we added and then removed.** The transactional-sender demotion at
  `0.55` costs 0.037 recall@5 [0.010, 0.072] and 0.045 MRR@10 [0.009, 0.085] and
  is significant on all five metrics: it demoted the mail people were searching
  for. See "Bounded priors" above.
- **BM25 alone lost its lead.** It is now 0.153 MRR@10 [0.058, 0.251] behind the
  hybrid; on the frozen corpus it led by 0.126. Quoted text moved to a column
  BM25 weights lower, which is exactly the text the dense leg still reaches.
- **Collapse is a trade you can now price.** It buys MRR@10 0.010 [0.002, 0.022]
  and takes queries whose top 10 repeats one body from 27 down to 12; it costs
  recall@10 0.058 [0.027, 0.093], nDCG@10 0.032 [0.009, 0.059] and P@5
  0.031 [0.012, 0.055]. The labels it removes are template families (twelve DPD
  delivery notices, every one labelled relevant), so if your answer is "every
  copy of this mail", run `--no-collapse`.
- **Not resolved: the prompt contract.** The `title: | text:` format against the
  old `Document:`/`SearchQuery:` one differs by 0.005 to 0.011 on every metric,
  all intervals across zero. It is the documented contract of the model, so it
  stays, but this eval cannot claim credit for it.
- **The largest single fix was a bug.** `Afgeleverd: je pakket van Lyon` parsed
  as mail *from* someone called Lyon and returned an empty list; four of the 84
  queries did. Guarding the parser (see "Filters as SQL") is worth 0.05 MRR@10
  and 0.06 recall@5, more than any tuning knob moved.
- **Where the misses moved.** The buckets went from 7 never retrieved and 16
  retrieved-but-buried to 3 and 7, with 60 of 70 now clean. What is left is a
  retrieval problem in three queries, not a ranking problem.

The corpus behind those numbers: 69 835 chunks to 47 720 (-32 %; the derived
windows alone went 65 861 to 34 362, -48 %), 565 messages whose subject was
stored as an RFC 2047 encoded-word down to 0, 45 756 vectors of chunks that no
longer existed pruned, 534 MB to 322 MB, and 1 918 ms to 1 241 ms per query.

What did **not** work is the relevance floor. Top scores of the out-of-corpus
queries run 0.379 to 0.762 (median 0.444) against 0.325 to 0.978 for queries that
do have an answer (median 0.656): the distributions sit on top of each other, so
no threshold tells "nothing here" from "found it". A floor of 0.45 would catch
8 of the 14 unanswerable queries and silence 15 of the 70 good ones; 0.30, the
default, catches nothing and silences nothing. `python floor_sweep.py` prints
that table again from a saved run. Saying "I found nothing" needs a signal that
does not exist in the fused score, probably a reranker's absolute score, and
pretending otherwise would cost real answers.

## Tests

```bash
python test_mailbox.py        # 216 tests, no endpoint and no network needed
python test_eval.py           # 88 tests for the evaluation tools
python test_gui.py            # 123 tests for the web layer, same stub endpoint
python test_mailhtml.py       # 36 tests for the mail view, no IMAP at all
```

The suites stub the embedding endpoint with a deterministic bag-of-words vector,
so they cover chunking invariants (no lost text, no duplicate windows), the
legacy-DB migration and `reindex --clean`, the FTS query builder against real
FTS5 (variants, prefixes, operator leaks), RRF arithmetic and leg weighting, the
bounded priors, thread collapse, the `queryparse` date/sender/recipient cases
(including the ones that must *not* parse: `van maart`, `per maand`), the
filter pre-filter, the `search_query()` contract, and both commands' CLI
plumbing. `test_embed_media.py` is the opposite: it hits your real endpoint and
reports what it can actually do.

Searching *with* a picture gets its own stub, because the shared one is a
text-only endpoint on purpose: `pixel_post` embeds a media item from the bytes
after `base64,` and nothing else, so "these two files look alike" is something
the suite can state instead of assume. On that stub the tests assert that a query
image goes out bare (no filename, no `title:` prefix, no text part, in either
envelope), that the same bytes indexed and queried are one vector, that a picture
finds the mail it came from, that one side cannot dilute the other
(`max`, not a mean: unrelated words leave the picture's score untouched, an
unrelated picture leaves the words' untouched), and that an endpoint which cannot
read pixels is refused in a sentence — native `/api/embed` before the request is
even made, `embed_media_multimodal: false` by name, a rejection after both
envelopes were tried — rather than answered with a text fallback that would rank
every mail saying "image attachment".

The reconcile tests run against a fake IMAP connection made of dicts, which is
the only way to reach the parts that actually matter: that a folder listing one
message *fewer* takes that message out of chunks, vectors and FTS together; that
a folder listing *nothing* takes nothing out; that a changed `UIDVALIDITY`
re-maps UIDs instead of deleting mail; that a run wanting more than
`prune_max_fraction` of a folder has to be forced; that a `--dry-run` leaves even
the UIDs it just learned unwritten; and that mail which comes back is re-fetched
by UID — the one path an incremental `SINCE` run can never take again. Two of
them test the contract rather than the arithmetic: one asserts nothing outside
`READ_ONLY_COMMANDS` was ever asked for (and trips on a read-write `SELECT`, a
`STORE`, an `EXPUNGE`), the other asserts the exact bytes that reach the server,
because a mailbox name `imaplib` has re-quoted is a `BAD` reply in production and
a green test right here. The reply direction deserves the same suspicion: `imaplib`
hands back a fetch item that carries no literal as one bare line of bytes, and only
an item with a literal as a `(prefix, literal)` pair, so a reader written against
pairs reads nothing from Gmail's answer — `X-GM-MSGID`, which carries no literal,
came back empty for every message in the mailbox while a fake that always handed
back pairs said everything was fine. `FakeIMAP` answers in the shape `imaplib`
really produces, in both directions. The two numbers the interface prints each have a test of
their own, because both are easy to say wrongly: *last dropped* is read from the
newest tombstone, so a run that removed nothing may not borrow it, and a UID the
folder lists with no row is counted as never fetched rather than mourned as gone.

The connection those tests run on is examined too, because its lifetime is what a
click waits on: that two files in one folder cost one login and one `EXAMINE`, that
a socket Gmail already closed is reopened *once* and the reconnect counted, that a
missing folder is an answer and not a broken pipe, and that the configured read
budget really does go on the socket — so a Gmail that stops in the middle of a
command costs a fresh login, not a request that never finishes and a worker thread
that never comes back with it.

The previews get the same treatment, against the same fake connection: that a
backfill asks for the one MIME part a picture lives in and never `RFC822`; that a
`--dry-run` issues *zero* IMAP commands and writes nothing; that a row with no UID
is reported as unreachable instead of guessed at; that `--clear` deletes every
preview and keeps every word of the index; that an orphaned preview is named before
it is offered for removal; and that `--links` writes only rows whose link is still
empty, again with nothing outside `EXAMINE`, `UID FETCH` and `LOGOUT` on the wire.
Reading one file out of a message is covered from both ends: that a
`BODYSTRUCTURE` reply is numbered the way IMAP numbers it (nested containers
included, `{n}` literals included, and a message that is one part having no
numbered parts at all), that every part a reply names is filed so the next one
costs a single command, that a part over the size budget is refused before its
bytes are asked for, and that a section which is simply not there *ends* the search.
That last one is asserted as the exact commands on the wire, because Gmail's answer
to a batch holding one missing section is `NO … could not be FETCHed` and nothing
else — the reply this path exists to survive, and the one a fake generous enough to
answer each section separately once waved through.

`test_gui.py` drives the routes with Starlette's test client over the same
stubbed endpoint. What it pins down is the parts a browser cannot tell you
about: that the index cache is invalidated by every write; that `page_messages`
and `filter_message_ids` still agree on the same `Filters` object (one filter
builder, two readers, and a chip that meant something else in the list pane
would be invisible); that dropping a chip removes the restriction *and* gives
the words back; that `SystemExit` from the CLI-shaped engine becomes a 409 with
a sentence in it instead of a dead server; that an unreachable endpoint degrades
to BM25 and says so; that opening an attachment asks Gmail only for read verbs,
keeps executable types (HTML, SVG) out of `inline`, and answers a gone message
with the same sentence the reading pane gives; that a nested `submit()` is
refused rather than hanging the one worker; that a picture in the query is a
search and not a parsing error, that bytes which are not a picture — or lie about
being one, or are over the cap — are refused with the reason in the sentence, that
an image search is *not* quietly degraded to BM25 when the endpoint is down
(there are no keywords to fall back to), and that no response echoes the image
bytes back; that a preview is served off disk with the IMAP connection never
opened, that it comes back smaller than the side says and with the size the pane
sizes its box from, that the browser is told to keep it and a revalidation answers
with an empty `304`, that a file with no preview is refused with the command that
cuts one, that a list row *and* a search result carry their picture's filename (one
sidecar query for the page, not one per row), that the status line counts the
previews kept, and that a dropped message loses its thumbnail along with its row;
and that **mail text never becomes markup** — a message whose
subject is `<img src=x onerror=alert(1)>` is served as JSON data and both front-end
scripts are asserted to contain none of the four words that would parse markup into
the page (`innerHTML` and its three relatives), in code or in prose, so that a grep
across `web/` is a complete answer. Both files are also run
through `node --check` when Node is installed (skipped when it is not), because
a stray parenthesis once left the page a silent static shell while every
Python-side assertion still passed. One more runs the front end's *arithmetic*
under Node: `fitSide` and `dataBytes` are lifted out of `web/app.js` and called,
so the size a photo is reduced to and the byte count the cap is judged against
are the shipped ones — which is how a `length * 3/4` that counted base64 padding
as payload turned up as a failing test instead of as a tooltip lying by a byte.

The mail view is tested in the two places its promises are actually made.
`test_mailhtml.py` holds the sanitizer to what mailboxes contain: a script tag, an
`onerror`, a conditional comment and an unclosed `<iframe>`; the head stylesheet
that survives while its `@import` and `@font-face` do not; `display:none` kept and
`position:fixed` refused; a `cid:` that becomes the message's own attachment URL and
a relative `src` that is refused outright; `srcset` missing from the allowlist on
purpose, because the library does not check schemes there; the two tag lists that
make a Rust library *panic* rather than raise if a tag lands in both; and the depth
guard, timed (100 000 nested divs is refused in well under a second, where the
sanitizer alone took 21.7). `test_gui.py` then holds the seam: the body arrives as
`application/json` and never as a document, the frame's CSP is written before the
mail's markup, the part on screen is the part named and not the part searched, a
`cid:` image is an ordinary attachment request, images stay hidden until the reader
asks and the CSP widens only for that answer, a mail in another charset is decoded by
the charset it declares (while the indexer's older rule stays put, so search results
do not move), a mail with no HTML part is a 404 carrying `reason: no_html` rather than
a sentence to pattern-match, the second look issues *no* IMAP command at all, nothing
outside `EXAMINE`/`UID SEARCH`/`UID FETCH`/`LOGOUT` is ever asked, no `UID FETCH` names
a body without `BODY.PEEK`, and the render runs on the media worker so `queued` keeps
meaning *searches waiting*. The frame itself is built by Node against a stubbed DOM
that records writes in order, which is the only way to assert that `sandbox` carries
exactly the two permissions it should, and gets them before the document exists.

## Notes

- A run is a **streaming pipeline**: it fetches and indexes in batches of 50
  (32 embeddings per request), so **recent mail is committed (and searchable)
  first** and the whole mailbox is never held in memory. Within each batch,
  **all message text (body + metadata) is embedded before any attachment is
  processed** — text-extracted attachments are batched, media attachments are
  embedded one at a time after the text phase. One `tqdm` progress bar tracks
  the run (total pre-counted with a cheap `UID SEARCH`, so it shows a real
  percentage and ETA); a per-batch log line shows progress.
- **Newest-first**: messages are fetched and indexed most-recent-first ("work
  back in time"), so a long `--full` sweep makes recent mail usable quickly.
- **Resumable**: a `--full` run stores a `full_run_watermark` (the oldest date
  committed so far) after each batch. If you interrupt it (Ctrl-C), re-run the
  same command and it resumes with `BEFORE <watermark + 1 day>`, skipping
  already-indexed mail via `Message-ID` dedup. On completion the watermark is
  cleared and `last_indexed_date` is set. `--limit` test runs and incremental
  runs don't use the watermark.
- Mail is fetched with `UID SEARCH` / `UID FETCH` (stable UIDs, not
  sequence numbers). The FETCH message set is sent **comma-separated**
  (`UID FETCH 4,5,6 (RFC822)`): Gmail rejects the standard space-separated
  multi-UID form with `BAD "Could not parse command"` but accepts the comma
  (and range) form. As a safety net, if a server still rejects a batched
  FETCH, the script retries that batch one message at a time, so the run
  still completes. Use `--debug` to see every IMAP byte if you need to
  diagnose a server error.
- **Read-only against the mailbox, by construction.** Folders are opened with
  `EXAMINE`, never `SELECT`; bodies come from `BODY.PEEK[…]`, so a FETCH cannot
  even mark a message read in Gmail; and `GmailFetcher.READ_ONLY_COMMANDS` lists
  every verb the code may issue — a test runs the reconcile against a fake
  connection that raises on anything else, including a read-write `SELECT`.
- Which brings a quirk of `imaplib` with it: `_command` quotes **every `str`
  argument**, so a mailbox name has to arrive pre-quoted (`b'"[Gmail]/All
  Mail"'`) and a SEARCH criterion or UID set has to arrive as **bytes** — hand
  either to the connection as a `str` and Gmail answers
  `BAD Could not parse command`. `GmailFetcher._bname()` and `_bytes()` are the
  two helpers that keep that straight; anything new that touches the connection
  should go through them (the reconcile tests assert the exact wire form).
- Dedup is by `Message-ID`; re-indexing a message **replaces** its chunks,
  vectors and FTS rows (chunks, embeddings and `chunks_fts` are deleted in that
  order, so no orphan row survives).
- `search` loads all vectors into memory once per run (~80 MB per 25 000 chunks
  at dim 768) and scores by brute force; `stats` reports the current vector
  bytes. `dense_pool` is the knob if that ever grows past a mailbox.
- The FTS index is kept in sync on every commit, and re-synced (gap-filled) on
  open, so an interrupted run cannot leave keyword search silently stale.
- The SMTP host/port are stored in config for future use; v1 is read-only.
- `mailbox.db` is the only state file. It upgrades in place, so an old index is
  worth keeping — `reindex --dry-run` shows what a tier change costs before you
  commit to it. Deleting it means re-fetching every message from Gmail.
