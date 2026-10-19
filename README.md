# Procurement evidence workspace

Private interview project for helping a supplier understand customer procurement requirements, inspect supporting evidence, track unresolved work and commitments, and prepare accurate customer updates.

**Implementation status:** planning and supplied inputs only. No application has been implemented or validated yet.

## Start here

The [main epic](https://github.com/akshay-rakheja/inperson-interview/issues/1) is the source of truth for the design, agreed assumptions, implementation sequence, prototype acceptance criteria, and later roadmap. Its child issues contain implementation contracts, source references, dependencies, and validation requirements for an agent continuing the work.

The current implementation budget is **4-5 hours**, superseding the original brief's two-day allowance. Follow the prototype milestone first; later work is explicitly out of that time box.

## Supplied materials

- [Interview brief](AI_Engineer_Procurement_Work_Interview_Clean%20%281%29.pdf)
- [Acme Bank case](CASE_1_Acme_Bank/00_START_HERE.txt)
- [Northstar Telecom case](CASE_2_Northstar_Telecom/00_START_HERE.txt)

Each case contains 14 original files, including a 60-row questionnaire. The cases share questionnaire wording but have different customer scope and review processes. Preserve the source files unchanged. The brief is assignment context, not evidence of a customer's procurement requirement.

## Intended prototype

One local Python/Streamlit application, SQLite persistence, and one configured LLM provider. The complete demonstration loop is:

`Select case -> inspect requirements and evidence -> draft an owner-reviewed customer update -> add new information -> review what changed`

The prototype will preserve provenance, separate proposals and assertions from verification, expose missing evidence and unconfirmed dates, and retain prior analysis and draft history. It will not claim that an evidence note is an attached report, a planned control is implemented, or supplier review equals customer approval.

## Continuing the work

1. Read the epic and select the first unblocked prototype child issue.
2. Read that issue's contracts and linked original inputs before editing.
3. Keep implementation notes, decisions, verification results, and follow-ups on the relevant GitHub issue. Avoid a parallel local planning tracker.
4. Record exactly what was tested and what remains uncertain. A synthetic fixture or a local review marker does not establish live customer validation or authenticated owner approval.
5. Close only work whose acceptance criteria are satisfied; leave incomplete work and dependencies open.

Keep model credentials in ignored local environment/configuration files. Runtime databases and generated drafts under `data/` are ignored. Future model requests must be explicit user actions and must not be triggered merely by interface rerenders.
