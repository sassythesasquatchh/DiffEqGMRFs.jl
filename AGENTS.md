# Agent Guidance for Rust Port

## Scope
These instructions apply to the entire repository for the Rust migration effort.

## Expectations
- Follow the roadmap in `RUST_PORT_PLAN.md` and keep Rust modules aligned with existing Julia functionality.
- Maintain feature parity first; avoid premature optimizations until tests match Julia outputs on small fixtures.
- Prefer idiomatic Rust (ownership, `Result` error handling, `Option` instead of sentinel values) and document mathematical formulas near their implementations.
- Keep Julia reference code intact for comparison while porting; add Rust code in a new workspace until replacement is verified.

## Testing & validation
- Add unit tests for each ported component (datasets, metrics, assemblers, SPDE) and integration tests mirroring the Julia scripts.
- Use reproducible seeds for stochastic pieces and compare numerical tolerances against Julia baselines.

## Documentation
- Update `RUST_PORT_PLAN.md` as milestones complete; include notes on dependency choices and any deviations from Julia behavior.
