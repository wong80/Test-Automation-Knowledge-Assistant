# Validation Guide — per BACKLOG.md ticket

For every ticket: the **milestone** (definition of done), the **test script** (file to
write or command to run), and **how to validate** (manual acceptance steps).

Conventions: tests live in `tests/test_*.py` using pytest + pytest-mock. Integration
tests that need real services are marked `@pytest.mark.integration` (run with
`pytest -m integration`). All commands run from the repo root after `uv sync`.

---

## Phase 0 — Scaffolding

### P0-1: Repo restructure

**Milestone:** Directory layout matches BACKLOG (`/corpus`, `/ingest`, `/eval`,
`/retrieval`, `/codegen`, `/app`, `/docs`); no logic changes; full test suite green.

**Test script:** `tests/test_restructure.py`
```python
EXPECTED_DIRS = ["corpus", "ingest", "eval", "retrieval", "codegen", "app", "docs"]

def test_directories_exist():
    for d in EXPECTED_DIRS:
        assert Path(d).is_dir()

def test_existing_tests_still_import():
    import app.main, ingest.run, app.search  # noqa
```
Run: `pytest tests/test_restructure.py -v && pytest -q` (entire suite must stay green).

**How to validate:**
1. `git status` — only moves, no content diffs in moved files.
2. `pytest -q` — zero failures vs. pre-move baseline.
3. Smoke-run app: `streamlit run app/main.py` opens without error.

### P0-2: Audit current pipeline

**Milestone:** `/docs/audit.md` exists and documents: chunker (type + params),
embedding model, vector store, retrieval method, hardcoded paths/configs — with no
"see source" dead ends.

**Test script:** `tests/test_audit_doc.py`
```python
def test_audit_mentions_required_topics():
    text = Path("docs/audit.md").read_text()
    for kw in ["chunk", "embedding", "vector", "retrieval", "config", "path"]:
        assert kw.lower() in text.lower()

def test_audit_cross_checks_source():
    # every code path named in the audit must exist in the repo
    for fname in re.findall(r"`([\w/]+\.py)`", text):
        assert Path(fname).exists()
```
Run: `pytest tests/test_audit_doc.py -v`.

**How to validate:** Fresh-reader test — hand `/docs/audit.md` to someone who hasn't
seen the code; they must be able to explain the pipeline and reproduce the ingest
commands from the doc alone.

### P0-3: sources.yaml schema

**Milestone:** `/corpus/sources.yaml` with schema `id, url, doc_type,
instrument_family, vendor, version, date_fetched, license_note`; validates via
`validate_sources.py`; PyVisa source listed.

**Test script:** `tests/test_sources_schema.py` (uses pydantic model shared with
`/ingest/validate_sources.py`)
```python
def test_existing_sources_valid():
    rows = load_sources("corpus/sources.yaml")     # from ingest.validate_sources
    assert len(rows) >= 1
    assert all(r.instrument_family and r.url for r in rows)

def test_required_fields_present():
    for r in load_sources("corpus/sources.yaml"):
        for f in ["id", "url", "doc_type", "vendor", "version",
                  "date_fetched", "license_note"]:
            assert getattr(r, f) is not None
```
Run: `python ingest/validate_sources.py corpus/sources.yaml` and
`pytest tests/test_sources_schema.py -v`.

**How to validate:** Add a row with a missing field → validator exits non-zero.
Remove a row → validator reports the missing source.

### P0-4: Provider-agnostic LLM interface

**Milestone:** `app/llm.py` calls `litellm.completion(model=LLM_MODEL, ...)`; `.env.example`
documents OpenAI/Anthropic/Ollama options; Phase 5 calls this wrapper (no hardcoded
client anywhere).

**Test script:** `tests/test_llm_provider.py`
```python
def test_wrapper_reads_model_from_env(mocker):
    mocker.patch.dict(os.environ, {"LLM_MODEL": "ollama/llama3"})
    mock = mocker.patch("app.llm.completion")
    app.llm.llm("ping")
    mock.assert_called_once()
    assert mock.call_args.kwargs["model"] == "ollama/llama3"

def test_ollama_no_api_key_needed():
    # env has no OPENAI/ANTHROPIC key; wrapper must not raise on construction
    assert app.llm.client_factory() is not None
```
Run: `pytest tests/test_llm_provider.py -v`.

**How to validate (the BACKLOG acceptance):**
1. `LLM_MODEL=ollama/llama3` with no API keys set → `python -m codegen.generate_snippet --query "set vertical scale"` runs end-to-end (Ollama running locally), zero cost.
2. Change only `LLM_MODEL=anthropic/claude-sonnet-4-6` + `ANTHROPIC_API_KEY` → same command works, no code edit.
3. `grep -rn "OpenAI()\|openai\." app codegen ingest` → no direct client usage.

---

## Phase 1 — Corpus expansion

### P1-1: Source new PDFs

**Milestone:** 4-6 public PDFs on disk in `/corpus/raw/`; `sources.yaml` entries
complete and valid; every entry's file exists.

**Test script:** `tests/test_corpus_files.py`
```python
def test_at_least_four_pdfs():
    pdfs = list(Path("corpus/raw").glob("*.pdf"))
    assert len(pdfs) >= 4

def test_sources_point_to_existing_files():
    for r in load_sources("corpus/sources.yaml"):
        assert (Path("corpus/raw") / r.id).with_suffix(".pdf").exists()

def test_pdfs_are_non_trivial():
    for p in Path("corpus/raw").glob("*.pdf"):
        assert p.stat().st_size > 50_000
```
Run: `pytest tests/test_corpus_files.py -v`.

**How to validate:** Open each PDF — renders, not a stub page. Confirm `license_note`
is filled (IEEE 488.2 is paywalled — if used, only legally obtainable copies).

### P1-2: PDF parsing pipeline

**Milestone:** `/ingest/parse_pdf.py` (pdfplumber) emits one JSONL per source into
`/corpus/parsed/`; each line tagged `source_id`, `page_num`; tables/command syntax
intact.

**Test script:** `tests/test_parse_pdf.py`
```python
def test_parses_every_raw_pdf(tmp_path):
    run_parse(Path("corpus/raw"), tmp_path)          # ingest.parse_pdf
    out = list(tmp_path.glob("*.jsonl"))
    assert len(out) == len(list(Path("corpus/raw").glob("*.pdf")))

def test_lines_have_required_tags():
    line = json.loads(first_line_of("corpus/parsed"))
    assert {"source_id", "page_num"} <= line.keys()
    assert isinstance(line["page_num"], int)

def test_line_count_sanity():
    # every source yields >= 5 pages of content
    for f in Path("corpus/parsed").glob("*.jsonl"):
        assert sum(1 for _ in open(f)) >= 5
```
Run: `python ingest/parse_pdf.py --all` then `pytest tests/test_parse_pdf.py -v`.

**How to validate:** Spot-check 3 pages per doc type (scope guide / spec / app note):
the command-syntax tables (e.g. `:MEASure:VAMPlitude?` tables) read correctly — no
columns merged or cells truncated.

### P1-3: Command taxonomy tagging

**Milestone:** regex extractor in `/ingest/` tags sections containing SCPI command
syntax with `command_family` in JSONL metadata.

**Test script:** `tests/test_taxonomy.py`
```python
KNOWN_COMMANDS = [":MEASure:VAMPlitude?", ":TRIGger:MODE", ":CHANnel1:SCALe"]  # 20 total

def test_recall_on_known_commands(parsed_corpus):
    found = find_commands(parsed_corpus)             # ingest.taxonomy
    recall = len(set(KNOWN_COMMANDS) & set(found)) / len(KNOWN_COMMANDS)
    assert recall >= 0.8

def test_tagged_lines_have_family():
    for line in tagged_lines("corpus/parsed"):
        if line.get("is_command_section"):
            assert "command_family" in line
```
Run: `pytest tests/test_taxonomy.py -v`.

**How to validate:** Manually scan the 20-command sample list — every command you know
is in the doc must appear in the extractor's output; false positives documented in
the script header.

---

## Phase 2 — Structure-aware chunking

### P2-1: Chunking boundary detector

**Milestone:** boundary-aware splitter (command blocks / section headers, not token
counts) → `/corpus/chunks/*.jsonl`; no mid-command or mid-sentence cuts.

**Test script:** `tests/test_chunk_boundaries.py`
```python
def test_no_mid_command_cuts(all_chunks):
    for c in all_chunks:
        assert not c["text"].startswith((":", ";"))        # never starts mid-command
        assert c["text"].rstrip().endswith((".", "?", "}", ")"))  # ends on boundary

def test_no_split_inside_command_block(chunk_texts):
    # a command block with its full syntax must stay in one chunk
    assert any(":MEASure" in t and "VAMPlitude?" in t for t in chunk_texts)

def test_chunks_cover_corpus():
    covered = sum(len(c["text"]) for c in all_chunks)
    assert covered >= 0.95 * total_parsed_text()
```
Run: `python ingest/chunk.py --all` then `pytest tests/test_chunk_boundaries.py -v`.

**How to validate:** Manually inspect 10 random chunks — none split a command syntax
block or start/end mid-sentence.

### P2-2: Chunk metadata enrichment

**Milestone:** every chunk carries `source_id, page_num, command_family (if
applicable), chunk_id (stable hash), instrument_family, doc_type`; no nulls.

**Test script:** `tests/test_chunk_schema.py`
```python
REQUIRED = ["source_id", "page_num", "chunk_id", "instrument_family", "doc_type"]

def test_required_fields_never_null(all_chunks):
    for c in all_chunks:
        for f in REQUIRED:
            assert c[f] is not None

def test_chunk_id_is_stable_hash():
    a, b = chunk_id(c["text"], c["source_id"]), chunk_id(c["text"], c["source_id"])
    assert a == b and len(a) == 16

def test_chunk_id_derives_from_content():
    # tiny change in text must change the id
    assert chunk_id("x") != chunk_id("y")
```
Run: `pytest tests/test_chunk_schema.py -v`.

**How to validate:** Run the schema check script over all chunks → zero nulls; re-run
chunking → identical `chunk_id`s (idempotent).

### P2-3: Re-embed and re-index

**Milestone:** new chunk set embedded and indexed (Qdrant or existing store); old
index preserved as `v0`.

**Test script:** `tests/test_index_smoke.py`
```python
def test_new_index_queryable():
    results = search("vertical scale 1.2V DDR5", method="hybrid")
    assert len(results) >= 5 and results[0]["score"] > 0

def test_v0_index_preserved():
    assert index_exists("v0") and index_exists("v1")

def test_query_returns_metadata():
    r = search("set trigger mode")[0]
    assert {"source_id", "chunk_id", "page_num"} <= r.keys()
```
Run: `pytest tests/test_index_smoke.py -v`.

**How to validate:** Run the 5 manual smoke queries from the BACKLOG — each returns
plausible top hits (relevant instrument doc on top, no off-topic results).

---

## Phase 3 — Retrieval evaluation harness (highest priority)

### P3-1: Write eval question set

**Milestone:** `/eval/questions.jsonl` with 30-50 questions, each with ≥1 verified
`relevant_chunk_id` (or `doc_id`), spread across instrument families.

**Test script:** `tests/test_questions.py`
```python
def test_at_least_30_questions():
    qs = load_questions("eval/questions.jsonl")
    assert len(qs) >= 30

def test_ground_truth_chunk_ids_resolve():
    ids = {c["chunk_id"] for c in all_chunks}
    for q in qs:
        assert any(i in ids for i in q["relevant_chunk_ids"])

def test_spread_across_families():
    fams = {q["instrument_family"] for q in qs}
    assert len(fams) >= 3   # not clustered on one doc
```
Run: `pytest tests/test_questions.py -v`.

**How to validate:** Manually open 5 questions and confirm the labeled chunk genuinely
answers the question (eyeball against `/corpus/chunks/`).

### P3-2: Eval metrics script

**Milestone:** `/eval/run_eval.py` computes hit-rate@k (k=1,3,5) + MRR for a given
retrieval config (CLI arg or config file); writes `/eval/results/*.json` + stdout.

**Test script:** `tests/test_run_eval.py`
```python
def test_metrics_on_toy_data():
    metrics = evaluate([[[True], [False]], [[False, True], [False, False]]])
    assert metrics["hit_rate@1"] == 0.5
    assert metrics["mrr"] == 0.75

def test_perfect_retrieval_scores_1():
    perfect = [[[True, True]] * 3]
    assert evaluate(perfect)["hit_rate@3"] == 1.0 and evaluate(perfect)["mrr"] == 1.0
```
CLI smoke: `python eval/run_eval.py --config eval/configs/dense.json` exits 0 and
writes `/eval/results/dense.json`.

**How to validate:** Run against the question set — metrics between 0 and 1, results
file contains the three k-values and MRR.

### P3-3: Baseline run

**Milestone:** baseline numbers recorded, committed, referenced in `/eval/README.md`
(which states which chunking/retrieval config produced them + methodology).

**Test script:** `tests/test_baseline_recorded.py`
```python
def test_baseline_results_exist():
    assert Path("eval/results/baseline.json").exists()

def test_baseline_reproducible():
    a = json.loads(Path("eval/results/baseline.json").read_text())
    b = run_eval("eval/configs/dense.json", seed=42)   # same seed
    assert a["hit_rate@5"] == b["hit_rate@5"] and a["mrr"] == b["mrr"]
```
Run: `pytest tests/test_baseline_recorded.py -v`.

**How to validate:** Re-run with the same seed → identical numbers. README explains
the methodology in ≥5 lines.

---

## Phase 4 — Hybrid retrieval

### P4-1: BM25 retriever

**Milestone:** sparse retrieval via `rank_bm25` over the same chunk set; standalone
script (`/retrieval/bm25.py`); passes eval harness as its own config.

**Test script:** `tests/test_bm25.py`
```python
def test_known_terms_rank_highest(bm25_index, sample_chunks):
    hits = bm25_search("vertical scale", bm25_index, sample_chunks, k=5)
    assert any("SCALe" in h["text"] for h in hits)

def test_empty_query_returns_nothing():
    assert bm25_search("", bm25_index, sample_chunks) == []
```
Run: `pytest tests/test_bm25.py -v`, then
`python eval/run_eval.py --config eval/configs/bm25.json` — record results.

**How to validate:** BM25-only eval numbers are saved; a query with a rare exact token
(e.g. `:MEASure:VAMPlitude?`) ranks the right chunk #1.

### P4-2: Reciprocal Rank Fusion

**Milestone:** `rrf(dense_ranks, bm25_ranks)` in `/retrieval/fusion.py`, unit-tested.

**Test script:** `tests/test_fusion.py` (toy example from BACKLOG: known rankings in,
known fused ranking out)
```python
def test_rrf_known_output():
    dense = ["a", "b", "c", "d"]
    bm25  = ["c", "a", "d", "b"]
    fused = rrf(dense, bm25, k=60)
    assert fused[0] == "a"      # rank 1+2 = 1/61 + 1/62 beats c's 1/63 + 1/61

def test_rrf_penalizes_low_rank():
    # item only in one list at rank 4 must rank below item in both lists
    assert rrf(["a", "b"], ["a"], k=60) == ["a", "b"]

def test_rrf_k_constant_nonzero():
    with pytest.raises(ZeroDivisionError):
        rrf(["a"], ["a"], k=0)
```
Run: `pytest tests/test_fusion.py -v`.

**How to validate:** Known-rankings-in/known-rankings-out cases pass; scores computed
as `1/(k+rank)`.

### P4-3: Hybrid eval comparison

**Milestone:** `/eval/README.md` results table (baseline vs BM25 vs hybrid) with a
one-line conclusion on the winner and why.

**Test script:** `tests/test_comparison_documented.py`
```python
def test_readme_has_all_three_configs():
    text = Path("eval/README.md").read_text()
    for cfg in ["baseline", "BM25", "hybrid"]:
        assert cfg.lower() in text.lower()

def test_results_files_match_table():
    for name in ["baseline", "bm25", "hybrid"]:
        assert Path(f"eval/results/{name}.json").exists()

def test_conclusion_present():
    assert "conclusion" in Path("eval/README.md").read_text().lower()
```
Run: `python eval/run_eval.py --config eval/configs/hybrid.json` then
`pytest tests/test_comparison_documented.py -v`.

**How to validate:** Table's numbers match the three results JSONs; conclusion states
the winning config and a plausible reason (e.g. "hybrid wins because SCPI mnemonics
are exact tokens BM25 captures, while synonyms need dense").

---

## Phase 5 — Code-generation layer

### P5-1: Few-shot PyVisa examples

**Milestone:** 2-3 canonical runnable PyVisa examples in `/codegen/examples/` (connect,
`query()`, `write()`; scope/DMM/AWG usage).

**Test script:** `tests/test_examples_compile.py`
```python
def test_examples_are_valid_python():
    for f in Path("codegen/examples").glob("*.py"):
        ast.parse(f.read_text())        # raises SyntaxError if invalid

def test_examples_use_rm_and_query_write():
    code = "\n".join(f.read_text() for f in Path("codegen/examples").glob("*.py"))
    assert "pyvisa" in code and "query(" in code and "write(" in code
```
Run: `pytest tests/test_examples_compile.py -v`; if `pyvisa-sim` is set up, run each
example against it.

**How to validate:** Each example executes against the sim backend (or at minimum
parses + reviews clean; run manually with a stub `ResourceManager`).

### P5-2: Snippet generation prompt

**Milestone:** `/codegen/generate_snippet.py` takes a retrieved chunk, prompts the LLM
via the P0-4 wrapper with the few-shot examples, emits structured JSON
(`code`, `explanation`).

**Test script:** `tests/test_generate_snippet.py` (mock the wrapper — no API call)
```python
def test_output_schema(mocker):
    mocker.patch("codegen.generate_snippet.llm",
                 return_value=('{"code": "instr.query(\":MEASure?\")", "explanation": "e"}', {}))
    out = generate_snippet(retrieved_chunk, examples)
    assert out["code"] and out["explanation"]

def test_generated_code_uses_retrieved_command(mocker):
    chunk = {"text": "use :CHANnel1:SCALe ..."}
    mocker.patch("codegen.generate_snippet.llm",
                 return_value=('{"code": "instr.write(\":CHANnel1:SCALe 0.5\")"}', {}))
    assert ":CHANnel1:SCALe" in generate_snippet(chunk, examples)["code"]
```
Real check (10 chosen commands): generate all, then
`python -c "import ast; [ast.parse(s['code']) for s in snippets]"`.

**How to validate:** For 10 manually chosen retrieved commands: snippets are
syntactically valid Python and contain the retrieved command string.

### P5-3 (stretch): pyvisa-sim validation

**Milestone:** `pyvisa-sim` backend configured; ≥5 generated snippets execute without
error; limitations documented.

**Test script:** `tests/test_sim_execution.py` (mark integration)
```python
@pytest.mark.integration
def test_generated_snippets_execute_against_sim(sim_backend):
    for snippet in chosen_snippets():
        with pyvisa.ResourceManager(sim_backend) as rm:
            exec(snippet["code"], {"rm": rm})       # raises if instrument call fails
```
Run: `pytest -m integration tests/test_sim_execution.py -v`.

**How to validate:** ≥5 snippets run without exception; README section states the sim
validates execution only, not response correctness.

---

## Phase 6 — README & packaging

### P6-1: Architecture diagram

**Milestone:** non-ASCII-art diagram (PNG/SVG) in `/docs/assets/` reflecting
corpus → chunking → hybrid retrieval → codegen → app.

**Test script:** `tests/test_assets.py`
```python
def test_diagram_exists_and_is_image():
    p = next(Path("docs/assets").glob("architecture.*"))
    assert p.suffix in (".png", ".svg") and p.stat().st_size > 10_000
```
Run: `pytest tests/test_assets.py -v`.

**How to validate:** Render at README width — every stage legible; matches the actual
pipeline (cross-check against `/docs/audit.md`).

### P6-2: README rewrite

**Milestone:** Structure per BACKLOG: pain point → diagram → eval numbers (P4-3) →
example query/output → setup. Setup verified fresh.

**Test script:** `tests/test_readme_structure.py`
```python
def test_readme_leads_with_pain_point():
    text = Path("README.md").read_text()
    assert text.index("manual") < text.index("architecture")  # problem first

def test_readme_mentions_eval_numbers():
    assert "hit" in Path("README.md").read_text().lower()

def test_setup_commands_listed():
    assert "uv sync" in Path("README.md").read_text()
```
Run: `pytest tests/test_readme_structure.py -v`.

**How to validate:** Stranger test — someone follows setup on a clean machine with no
prior context and succeeds. First 3 paragraphs convey purpose + results.

### P6-3: Demo script/notebook

**Milestone:** one end-to-end artifact: question in → retrieved chunk → generated
snippet out, runs top-to-bottom unattended.

**Test script:** `tests/test_demo.py`
```python
def test_demo_runs_unattended():
    out = runpy.run_path("demo/run_demo.py")      # or subprocess
    assert out["snippet"]["code"] and "pyvisa" in out["snippet"]["code"]
```
For a notebook: `jupyter nbconvert --to notebook --execute --inplace demo/demo.ipynb`
(run in CI smoke).

**How to validate:** Execute cleanly with no manual intervention; output shows a real
question, its retrieved chunk source, and a runnable snippet.

---

## Phase 7 — MLOps Zoomcamp bridge (deferred)

### P7-1: MLflow tracking for retrieval configs

**Milestone:** eval runs logged to MLflow: params (chunk size, embedding model, fusion
weights) + metrics (hit-rate@k, MRR).

**Test script:** `tests/test_mlflow_tracking.py`
```python
def test_run_logged_with_params_and_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")
    with track_eval(config) as run:
        pass
    data = mlflow.get_run(run.info.run_id)
    assert "chunk_size" in data.data.params
    assert "hit_rate@5" in data.data.metrics
```
Run: `pytest tests/test_mlflow_tracking.py -v`.

**How to validate:** `mlflow ui` shows one run per eval config with params + metrics.

### P7-2: CI eval on corpus/retrieval changes

**Milestone:** GitHub Actions workflow re-runs eval on PRs touching `/corpus`,
`/retrieval`, `/ingest`; fails on regression beyond threshold.

**Test script:** `.github/workflows/eval.yml` + `eval/check_regression.py`
```python
def check_regression(baseline, new, threshold=0.05):
    return new["hit_rate@5"] >= baseline["hit_rate@5"] - threshold
```
Run locally: `python eval/check_regression.py baseline.json hybrid.json`.

**How to validate:** Open a PR touching `/corpus` → workflow runs and passes. Introduce
a junk corpus change → workflow fails with the regression diff printed.

### P7-3: Containerize retrieval + codegen service

**Milestone:** Dockerfile + docker-compose for retrieval API + codegen endpoint
(packaged, not deployed).

**Test script:** `tests/test_docker.py` (extend existing)
```python
def test_compose_builds():
    subprocess.run(["docker", "compose", "build"], check=True)

def test_healthcheck():
    for url in ["http://localhost:8000/health"]:
        assert requests.get(url, timeout=10).ok
```
Run: `docker compose build && docker compose up -d && pytest tests/test_docker.py -v`.

**How to validate:** `docker compose up` brings both services up; `/health` and
`/generate` respond; `docker compose down` cleans up.

---

## Test run matrix (quick reference)

| Phase | Run | Time |
|---|---|---|
| 0 | `pytest tests/test_restructure.py tests/test_audit_doc.py tests/test_sources_schema.py tests/test_llm_provider.py -v` | fast |
| 1 | `python ingest/parse_pdf.py --all && pytest tests/test_corpus_files.py tests/test_parse_pdf.py tests/test_taxonomy.py -v` | slow (parse) |
| 2 | `python ingest/chunk.py --all && pytest tests/test_chunk_boundaries.py tests/test_chunk_schema.py tests/test_index_smoke.py -v` | medium |
| 3 | `pytest tests/test_questions.py tests/test_run_eval.py -v && python eval/run_eval.py --config eval/configs/dense.json` | medium |
| 4 | `pytest tests/test_bm25.py tests/test_fusion.py -v && python eval/run_eval.py --config eval/configs/hybrid.json` | medium |
| 5 | `pytest tests/test_examples_compile.py tests/test_generate_snippet.py -v` | fast (mocked LLM) |
| 6 | `pytest tests/test_assets.py tests/test_readme_structure.py tests/test_demo.py -v` | fast |
| 7 | `pytest -m integration` + docker/mlflow manual steps | slow |

Full gate before any commit: `pytest -q` (unit) and, after Phase 3, the phase-3/4 eval
commands so the committed numbers stay reproducible.
