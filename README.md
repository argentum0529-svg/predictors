# Predictors

Self-contained project. It does not hold a Zerodha access token. Market data is read from the brokers service (`127.0.0.1:8785`), which keeps the token.

On the brokers host the client curls loopback. Anywhere else it ssh's to that host (`ZerodhaBroker.HOST`, default `argentum`) and curls loopback there.

```python
from datetime import date
from brokers import BaseBroker

broker = BaseBroker.default_broker("nse")  # ZerodhaBroker
broker.exchanges()                         # ["nse"]
candles = broker.fetch_minute("HAL", date(2026, 10, 1), date(2026, 10, 10))
# [[ts, open, high, low, close, volume], ...]
```

PA1 (what the baseline is, and how other predictors are compared to it): [PA1.md](PA1.md).
Design spec: `mpt/research/defense_entry_score.md`.
