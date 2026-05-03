# API Documentation

## Endpoints

### POST /ask
Ask a question to the RAG system.

**Request:**
```json
{
  "question": "What is the transformer architecture?"
}
```

**Response:**
```json
{
  "answer": "The transformer architecture...",
  "documents": [...],
  "source_type": "local_pdf"
}
```

## Response Codes

- `200`: Successful query
- `400`: Bad request
- `500`: Server error

## Examples

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is attention?"}
)
print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is attention?"}'
```
