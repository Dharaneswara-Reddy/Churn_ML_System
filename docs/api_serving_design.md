# API Serving Design

The churn model is exposed through a FastAPI service.

## Endpoint

POST /predict

The API:
1. Receives raw JSON input
2. Validates schema
3. Runs model inference
4. Returns probability and decision

## Design Choices

- Model loaded once at startup
- Training logic separated from inference(serving)
- Fail-fast validation strategy
- Stateless prediction requests 

### Stateless prediction requests

Each request is self-sufficient, the server does not store client session data, allowin for easier horizontal scaling.This is crucial for serverless architecures (eg: AWS Lambda) where instances may not persist between requests

---

The API never loads models from experiment folders and relies only on production artifacts.
