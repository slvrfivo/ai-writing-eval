from __future__ import annotations

import unittest

from src.llm.prompting import (
    ESSAY_TEXT_PLACEHOLDER,
    PROMPT_TEXT_PLACEHOLDER,
    build_messages,
    load_prompt_snapshot,
    render_user_prompt,
)


EXPECTED_SNAPSHOT_SHA256 = (
    "63684fd1d584f73d93eb91e2c00e8b0a027322112fd69a65c985771aa91403b0"
)


class PromptSnapshotTests(unittest.TestCase):
    def test_placeholders_occur_exactly_once(self) -> None:
        snapshot = load_prompt_snapshot()
        self.assertEqual(snapshot.user_template.count(PROMPT_TEXT_PLACEHOLDER), 1)
        self.assertEqual(snapshot.user_template.count(ESSAY_TEXT_PLACEHOLDER), 1)

    def test_rendering_replaces_both_placeholders(self) -> None:
        snapshot = load_prompt_snapshot()
        rendered = render_user_prompt(
            snapshot.user_template,
            prompt_text="주제 원문",
            essay_text="논증적 글 원문",
        )
        self.assertIn("[prompt_text]\n주제 원문", rendered)
        self.assertIn("[essay_text]\n논증적 글 원문", rendered)
        self.assertNotIn(PROMPT_TEXT_PLACEHOLDER, rendered)
        self.assertNotIn(ESSAY_TEXT_PLACEHOLDER, rendered)

    def test_snapshot_hash_is_pinned(self) -> None:
        snapshot = load_prompt_snapshot()
        self.assertEqual(snapshot.sha256, EXPECTED_SNAPSHOT_SHA256)
        self.assertEqual(snapshot.metadata["snapshot_sha256"], EXPECTED_SNAPSHOT_SHA256)

    def test_build_messages_uses_system_and_user_roles(self) -> None:
        messages = build_messages("주제", "본문")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("너는 한국어 논증적 글", messages[0]["content"])
        self.assertIn("[prompt_text]\n주제", messages[1]["content"])
        self.assertIn("[essay_text]\n본문", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
