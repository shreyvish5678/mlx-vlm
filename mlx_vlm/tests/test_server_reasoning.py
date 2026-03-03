"""Reasoning extraction tests for OpenAI-compatible chat responses."""

from mlx_vlm.server import _split_reasoning_content


def test_split_reasoning_content_with_explicit_think_tags():
    content, reasoning = _split_reasoning_content(
        "<think>Reason step 1.\nReason step 2.</think>\n\nFinal answer."
    )

    assert reasoning == "Reason step 1.\nReason step 2."
    assert content == "Final answer."


def test_split_reasoning_content_with_orphan_closing_think_tag():
    content, reasoning = _split_reasoning_content(
        "Reason step 1.\nReason step 2.</think>\n\nFinal answer."
    )

    assert reasoning == "Reason step 1.\nReason step 2."
    assert content == "Final answer."


def test_split_reasoning_content_without_thinking_tags():
    content, reasoning = _split_reasoning_content("No hidden reasoning here.")

    assert reasoning is None
    assert content == "No hidden reasoning here."


def test_split_reasoning_content_with_alt_thought_tokens():
    content, reasoning = _split_reasoning_content(
        "<|begin_of_thought|>Internal chain<|end_of_thought|>Visible answer"
    )

    assert reasoning == "Internal chain"
    assert content == "Visible answer"
