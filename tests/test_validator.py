"""Unit tests for JSON schema validation and retry loop."""

from __future__ import annotations

import pytest

from local_llm_router.config import ValidatorConfig
from local_llm_router.exceptions import MaxRetriesExceededError
from local_llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from local_llm_router.validator.loop import StructuredOutputLoop
from local_llm_router.validator.schema import (
    extract_json_candidate,
    parse_and_validate_json,
)

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "age"],
}


def test_extract_json_candidate():
    # Markdown wrapped
    md = "Sure, here is your json:\n```json\n{\"name\": \"Alice\", \"age\": 30}\n```\nEnjoy!"
    candidate = extract_json_candidate(md)
    assert candidate == '{"name": "Alice", "age": 30}'

    # Raw JSON
    raw = '{"name": "Bob", "age": 25}'
    assert extract_json_candidate(raw) == raw

    # Wrapped in plain text without codeblock
    text = 'Here is the data: {"name": "Charlie", "age": 40} Thank you.'
    assert extract_json_candidate(text) == '{"name": "Charlie", "age": 40}'


def test_parse_and_validate_json():
    # Valid
    valid_text = '{"name": "Alice", "age": 30}'
    ok, data, errors = parse_and_validate_json(valid_text, SAMPLE_SCHEMA)
    assert ok is True
    assert data["name"] == "Alice"
    assert len(errors) == 0

    # Syntax Error
    bad_syntax = '{"name": "Alice", "age": }'
    ok, data, errors = parse_and_validate_json(bad_syntax, SAMPLE_SCHEMA)
    assert ok is False
    assert any("syntax error" in e.lower() for e in errors)

    # Schema Validation Error (missing 'age')
    missing_field = '{"name": "Alice"}'
    ok, data, errors = parse_and_validate_json(missing_field, SAMPLE_SCHEMA)
    assert ok is False
    assert any("'age' is a required property" in e for e in errors)


@pytest.mark.asyncio
async def test_structured_output_retry_success():
    validator = StructuredOutputLoop(ValidatorConfig(enabled=True, max_retries=2))

    call_count = 0
    async def mock_execute(req: ChatCompletionRequest) -> ChatCompletionResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First turn: returns invalid JSON (missing age)
            return ChatCompletionResponse.create_simple(
                content='{"name": "Alice"}',
                model="test",
            )
        else:
            # Second turn: model corrected its output based on feedback
            assert len(req.messages) > 1
            assert "The previous response is invalid" in req.messages[-1].content
            return ChatCompletionResponse.create_simple(
                content='{"name": "Alice", "age": 30}',
                model="test",
            )

    req = ChatCompletionRequest(
        model="test",
        messages=[ChatMessage(role="user", content="Extract user info")],
        json_schema=SAMPLE_SCHEMA,
    )

    resp = await validator.execute_with_validation(req, mock_execute)
    assert resp.choices[0].message.content == '{"name": "Alice", "age": 30}'
    assert resp.router_metadata.validation_retries == 1
    assert call_count == 2


@pytest.mark.asyncio
async def test_structured_output_max_retries_exceeded():
    validator = StructuredOutputLoop(ValidatorConfig(enabled=True, max_retries=2))

    async def always_bad_execute(req: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse.create_simple(content="Not JSON at all", model="test")

    req = ChatCompletionRequest(
        model="test",
        messages=[ChatMessage(role="user", content="Extract user info")],
        json_schema=SAMPLE_SCHEMA,
    )

    with pytest.raises(MaxRetriesExceededError) as exc_info:
        await validator.execute_with_validation(req, always_bad_execute)

    assert exc_info.value.attempts == 2
