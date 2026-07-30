# Source Document Review Matrix

Status: Draft bootstrap.

This matrix is the intake format for large design material. It is intended for
the existing 400,000-token platform design document, the reduced 2-week MVP
document, PCX SRS material, and NeX-PCX commit history. The goal is to distill
requirements and decisions, not to copy source text wholesale.

## Review Columns

| Column | Description |
| --- | --- |
| Source document | File name, version, or source identifier. |
| Source section | Heading, page, anchor, or commit reference. |
| Claim or requirement | Concise statement extracted from the source. |
| PCX evidence | Related PCX implementation, test, smoke evidence, screenshot, or commit. |
| Target service | `nex-cx`, `nex-ae-web`, `nex-ae-back`, `nex-mo`, `nex-oa`, `nex-ag`, or shared. |
| MVP/defer | `MVP`, `Deferred`, `Rejected`, `Duplicate`, or `Needs Review`. |
| Design impact | SRS, architecture, data model, API, UI, operations, testing, or common module. |
| Open question | What must be decided before implementation. |
| Decision | Accepted decision and rationale. |
| Owner/status | Current owner and review status. |

## Matrix Template

| Source document | Source section | Claim or requirement | PCX evidence | Target service | MVP/defer | Design impact | Open question | Decision | Owner/status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400,000-token design document | TBD | TBD | TBD | TBD | Needs Review | TBD | TBD | TBD | Open |
| 2-week MVP document | TBD | TBD | TBD | TBD | Needs Review | TBD | TBD | TBD | Open |
| PCX SRS | TBD | TBD | TBD | TBD | Needs Review | TBD | TBD | TBD | Open |
| NeX-PCX commit history | TBD | TBD | TBD | TBD | Needs Review | TBD | TBD | TBD | Open |

## Review Rules

- One row should contain one decision-sized idea.
- Do not promote a broad source claim to MVP until an owner service is known.
- Keep implementation evidence separate from future preference.
- Mark duplicated large-document content as `Duplicate` instead of carrying it forward.
- Mark attractive but nonessential capabilities as `Deferred`.
- When the 400,000-token document conflicts with the 2-week MVP document, record
  the conflict and choose the smaller MVP unless the user explicitly expands scope.

## Initial Review Batches

| Batch | Scope | Output |
| --- | --- | --- |
| Batch 1 | 2-week MVP document | MVP capability shortlist and missing platform spine. |
| Batch 2 | PCX SRS and slice history | Evidence-backed requirements and deferred PCX features. |
| Batch 3 | 400,000-token design document | Architecture ideas filtered into MVP/deferred/rejected rows. |
| Batch 4 | Consolidation | First NeX-Platform SRS draft and service-level backlog. |
