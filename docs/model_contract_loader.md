# Model Contract Loader

## Problem
Inference validation read metadata.json on every API request,
causing repeated disk I/O.

## Solution
A cached Model Contract Loader was introduced.

At API startup:
- metadata is loaded once
- feature schema cached in memory

## Benefits
- Faster inference latency
- Reduced disk access
- Production-grade serving pattern