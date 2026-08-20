# ADR-0001: Record architecture decisions

- **Status:** accepted
- **Date:** 2026-08-20

## Context

The capstone hand-in requires key Architecture Decision Records, and the team
needs a lightweight way to remember why choices were made.

## Decision

Keep ADRs as numbered Markdown files in `docs/adr/`, using `0000-template.md`.
Every significant choice (stores, chunking strategy, reranking approach,
deployment shape) gets one.

## Consequences

Small ongoing writing cost; big payoff at handover and in the demo Q&A.
