# Procurement evidence workspace

Private interview project for helping a supplier understand customer procurement requirements, inspect supporting evidence, track unresolved work and commitments, and prepare accurate customer updates.

**Implementation status:** working local prototype for the 4-5 hour milestone (issues #2-#7), plus incremental reanalysis from #14. Exact verification results, live-model checks and remaining limitations are recorded on the GitHub issues, not here. Nothing in the app records authenticated approval, customer acceptance, delivery or deployment.

## How it works

`Case folder → deterministic import with source IDs → one structured model call → code checks → SQLite history → Streamlit workspace`

- **The model interprets.** One schema-constrained call reads the whole case and returns requirements, position statements (proposed / reported / historical / documented, with customer scope), gaps, explicit vs inferred dependencies, source-reported commitments and next actions. Drafting is a separate call that you trigger yourself.
- **Code checks and coordinates.**
  - Every citation must name a real source in this case, real lines and a quote that appears there; failures are removed and shown as gaps.
  - Runs are keyed on sources, date, model and prompt version, and a failed run never replaces a good one.
  - Owner action packs and run comparisons are plain grouping, with no model call.
  - New responses mark drafts stale, and reanalysis only regenerates the requirements a response is linked to.
- **People decide.** What is true, what can be shared and who owns what. Reviews are self-reported, and nothing is sent anywhere.
- **Guiding rules.** A plan is not proof, a claim is not evidence, and another customer's answer is not ours. Unknown dates stay unknown.

Known limits: text inputs only (no PDF or email); small cases (full-context analysis, about $0.20 and 2 minutes per run); a valid citation does not prove the conclusion is right; reviewer identity is not authenticated.

## Start here

The [main epic](https://github.com/akshay-rakheja/inperson-interview/issues/1) is the source of truth for the design, agreed assumptions, implementation sequence, prototype acceptance criteria, and later roadmap. Its child issues contain implementation contracts, verification records, and follow-ups.

## Setup and launch

Requires Python 3.11+. From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m streamlit run app.py
```

(`python -m venv .venv && .venv/bin/pip install -e ".[dev]"` works too.)

### Model configuration (OpenRouter)

- Paste your OpenRouter key after `OPENROUTER_API_KEY=` in the **project-root `.env`** file (ignored by Git). The app creates this file from a blank template on first start if it is missing, and never overwrites an existing one. `.env.example` shows the expected variables.
- Change the model by editing `PROCUREMENT_MODEL` (initial value `anthropic/claude-sonnet-5.5`) and restarting the app.
- Process environment variables take precedence over `.env`. `OPENAI_API_KEY` is not used.
- Requests go only to `https://openrouter.ai/api/v1` using the OpenAI-compatible SDK with strict JSON-schema structured outputs. There is **no automatic model or provider fallback**: failures are shown and the previous successful result is kept.
- With a blank key, the app still starts, shows where to paste the key, and disables analysis/drafting. Starting the app or rerendering it never makes a model request; only the **Run analysis**, **Reanalyse** and **Generate customer update** buttons do.

Optional command-line helpers (same store and adapter):

```bash
.venv/bin/python -m procurement.smoke                                  # one small authenticated schema check
.venv/bin/python -m procurement.cli analyze case_1_acme_bank 2026-10-19  # explicit analysis run, persisted to data/
```

### Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use injected fake model clients and never call the provider. They check provenance, case isolation, persistence, state handling and adapter error mapping, **not** semantic correctness of real model output.

## Demo walkthrough

1. Choose **CASE_1_Acme_Bank** and the assessment date in the sidebar. Click **Run analysis**, or reopen the saved run. An identical saved run is reused without a request.
2. **Requirements & gaps**: open a requirement to see its original questions and statuses, the position statements (proposed/reported/historical/documented, scope), cited source lines, gaps, dependencies (explicit vs inferred) and source-reported commitments. Open **Actions by suggested owner** for copyable consolidated requests.
3. **Customer update**: pick entries under **Requirements to include** and click **Generate customer update**. Each factual sentence carries a `[C#]` claim marker with checked citations. Edit the draft, record a self-reported local review, and download it if needed.
4. **New information**: load a demo fixture, e.g. *Engineering pen-test remediation status and retest commitment (simulated)*. Save it as a new source; existing drafts become stale. Click **Reanalyse with current sources**, then inspect **What changed** and the refreshed owner requests.
5. Switch to **CASE_2_Northstar_Telecom** and repeat with the simulated pilot-verification fixture.

Demo fixtures in `tests/fixtures/` are clearly labelled **simulated** inputs. They are stored as separate response sources and are never written into the original case folders.

## Supplied materials

- [Interview brief](AI_Engineer_Procurement_Work_Interview_Clean%20%281%29.pdf)
- [Acme Bank case](CASE_1_Acme_Bank/00_START_HERE.txt)
- [Northstar Telecom case](CASE_2_Northstar_Telecom/00_START_HERE.txt)

Each case contains 14 original files, including a 60-row questionnaire. Preserve the source files unchanged. The brief is assignment context, not evidence of a customer's procurement requirement.

## Code map

| File | Responsibility |
| --- | --- |
| `app.py` | Thin Streamlit entry point; explicit `.env` loading |
| `procurement/config.py` | `.env` handling and settings (OpenRouter only) |
| `procurement/ingest.py` | Case discovery, canonical line anchors, source IDs, raw questionnaire rows |
| `procurement/models.py` | Pydantic contracts (`SourceRecord`, `EvidenceRef`, `CaseAnalysis`, `CustomerUpdate`, ...) |
| `procurement/store.py` | SQLite: cases, sources, analysis_runs, reviews, drafts |
| `procurement/llm.py` | OpenRouter adapter, strict JSON schema, error mapping |
| `procurement/analysis.py` | Analysis prompt, citation/scope validation, cached explicit runs, incremental reanalysis |
| `procurement/actions.py` | Deterministic owner action packs (no model call) |
| `procurement/drafting.py` | Customer-update drafting, claim checks, edits, local review |
| `procurement/updates.py` | Pasted responses, draft invalidation, run comparison, review recheck |
| `procurement/ui.py` | Workspace views |

## Continuing the work

1. Read the epic and select the first unblocked issue.
2. Read that issue's contracts and linked original inputs before editing.
3. Keep implementation notes, decisions, verification results, and follow-ups on the relevant GitHub issue. Avoid a parallel local planning tracker.
4. Record exactly what was tested and what remains uncertain. A synthetic fixture or a local review marker does not establish live customer validation or authenticated owner approval.
5. Close only work whose acceptance criteria are satisfied; leave incomplete work and dependencies open.

Runtime databases and generated drafts under `data/` are ignored by Git.
