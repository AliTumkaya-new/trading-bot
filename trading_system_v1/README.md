# Trading System v1

This repository is the initial foundation for a **test-phase trading infrastructure** covering:
- Crypto (primary: Binance market data / test environments)
- Borsa Istanbul (research-grade adapter via Yahoo Finance / optional professional adapter later)

## Goals
- Shared core engine for multiple markets
- Broad scanner + signal engine
- Risk-managed paper trading
- Backtest-ready architecture
- Clean path to live integration later

## Current status
This is the **foundation layer**:
- Core models
- Abstract market data interface
- Binance market data adapter
- BIST research adapter
- Scanner engine
- Example momentum breakout strategy
- Risk rules
- Paper broker
- Simple runner

## Important notes
- `Yahoo Finance / yfinance` should be treated as **research-only**.
- Real-money trading is intentionally **not enabled** in this version.
- Add real execution only after scanner, risk, logging, and monitoring are fully validated.

## Suggested structure
```
src/
  core/
  data/
  strategies/
  risk/
  execution/
  engine/
  utils/
```

## Next steps
1. Add persistent storage (SQLite/Postgres)
2. Add structured logs and metrics
3. Add backtesting engine
4. Add portfolio constraints
5. Add exchange-specific execution adapters
6. Add dashboard/API layer
