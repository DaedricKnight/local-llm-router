"""Auto-retry validation loop with targeted LLM feedback on JSON schema errors."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from local_llm_router.config import ValidatorConfig
from local_llm_router.exceptions import MaxRetriesExceededError
from local_llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from local_llm_router.validator.schema import parse_and_validate_json

logger = logging.getLogger(__name__)


class StructuredOutputLoop:
    """Manages iterative retry loop for schema adherence."""

    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config

    async def execute_with_validation(
        self,
        request: ChatCompletionRequest,
        execute_fn: Callable[[ChatCompletionRequest], Awaitable[ChatCompletionResponse]],
    ) -> ChatCompletionResponse:
        """Run request through execution and retry loop if JSON schema validation fails."""
        schema = request.get_effective_schema()

        # If validation disabled or no schema requested, proceed without loop
        if not self.config.enabled or not schema:
            return await execute_fn(request)

        max_retries = request.max_retries if request.max_retries is not None else self.config.max_retries
        current_request = request
        last_errors = []
        last_raw_output = ""

        total_retries = 0

        for attempt in range(max_retries + 1):
            response = await execute_fn(current_request)
            content = response.choices[0].message.content if response.choices else ""
            last_raw_output = content

            is_valid, parsed_data, errors = parse_and_validate_json(content, schema)
            if is_valid:
                if total_retries > 0:
                    logger.info("JSON schema validation succeeded after %d retries", total_retries)
                if response.router_metadata:
                    response.router_metadata.validation_retries = total_retries
                else:
                    from local_llm_router.models import RouterMetadata
                    response.router_metadata = RouterMetadata(
                        provider="validator",
                        model_used=response.model,
                        validation_retries=total_retries,
                    )
                return response

            last_errors = errors
            total_retries = attempt + 1

            if attempt < max_retries:
                logger.warning(
                    "JSON validation failed (attempt %d/%d): %s. Triggering correction turn.",
                    attempt + 1,
                    max_retries,
                    "; ".join(errors),
                )

                feedback_text = (
                    "The previous response is invalid according to the required JSON schema.\n"
                    "Validation Errors:\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n\nPlease rectify the JSON output to strictly conform to the schema.\n"
                    "Return ONLY the valid JSON object with no markdown explanations or conversational text."
                )

                # Clone messages and append assistant response + user feedback
                updated_messages = list(current_request.messages)
                updated_messages.append(ChatMessage(role="assistant", content=content))
                updated_messages.append(ChatMessage(role="user", content=feedback_text))

                current_request = current_request.model_copy(
                    update={"messages": updated_messages},
                    deep=True,
                )

        raise MaxRetriesExceededError(
            attempts=max_retries,
            last_raw_output=last_raw_output,
            last_errors=last_errors,
        )
