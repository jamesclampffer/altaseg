# Masker tests

## Running

These need work.

```
python -m pytest tests -m "not integration"   # fast, doesn't run forward passes
python -m pytest tests                        # everything. requires connected gpu
```