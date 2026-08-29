"""
Pseudonymisation tests.

``subject_key`` is the boundary that lets prediction history be joined to a
customer for labelling and GDPR erasure without the event store holding a
directly identifying value. Two defects made that claim false:

* The salt defaulted to the literal ``"churn-default-salt"``, which is published
  in this open-source repository. Customer identifiers are a small enumerable
  space, so the whole ``subject_key`` column was reversible by brute force.
* No Unicode normalisation, so ``José`` submitted as NFC at prediction time and
  NFD at erasure time produced different keys — erasure silently matched nothing
  while reporting success.
"""

from __future__ import annotations

import unicodedata

import pytest

from churn_system.events.predictions import SubjectSaltMissingError, subject_key


class TestFailsClosed:
    def test_missing_salt_raises_rather_than_using_a_default(self, monkeypatch):
        """A published default salt provides no pseudonymisation at all."""
        monkeypatch.delenv("CHURN_SUBJECT_KEY_SALT", raising=False)
        monkeypatch.delenv("CHURN_ALLOW_UNSALTED_SUBJECT_KEYS", raising=False)

        with pytest.raises(SubjectSaltMissingError):
            subject_key("customer-1")

    def test_explicit_opt_out_disables_pseudonymisation(self, monkeypatch):
        """Running without subject keys must be a deliberate choice, not a fallback."""
        monkeypatch.delenv("CHURN_SUBJECT_KEY_SALT", raising=False)
        monkeypatch.setenv("CHURN_ALLOW_UNSALTED_SUBJECT_KEYS", "1")

        assert subject_key("customer-1") is None

    def test_no_hardcoded_salt_default_in_executable_code(self):
        """
        Guard against a default salt being reintroduced.

        Scans string *constants* via the AST rather than raw text, so the
        docstring that explains why the default was removed does not trip it.
        """
        import ast
        from pathlib import Path

        import churn_system.events.predictions as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

        # Collect docstring nodes so they can be excluded.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))

        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ]

        assert "churn-default-salt" not in literals

        # And no os.environ.get for the salt may carry a fallback value.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) == 2
            ):
                first = node.args[0]
                name = getattr(first, "id", getattr(first, "value", None))
                if name == "SUBJECT_SALT_ENV":
                    fallback = node.args[1]
                    assert (
                        isinstance(fallback, ast.Constant) and fallback.value == ""
                    ), "the salt lookup must not carry a usable default"


class TestKeyDerivation:
    def test_configured_salt_produces_a_key(self, monkeypatch):
        monkeypatch.setenv("CHURN_SUBJECT_KEY_SALT", "a-real-secret")
        key = subject_key("customer-1")

        assert isinstance(key, str)
        assert key != "customer-1"
        assert len(key) == 64

    def test_same_input_and_salt_is_stable(self, monkeypatch):
        """Erasure depends on the key being reproducible across processes."""
        monkeypatch.setenv("CHURN_SUBJECT_KEY_SALT", "a-real-secret")

        assert subject_key("customer-1") == subject_key("customer-1")

    def test_different_salts_produce_different_keys(self, monkeypatch):
        monkeypatch.setenv("CHURN_SUBJECT_KEY_SALT", "salt-one")
        first = subject_key("customer-1")
        monkeypatch.setenv("CHURN_SUBJECT_KEY_SALT", "salt-two")
        second = subject_key("customer-1")

        assert first != second

    def test_blank_identifier_yields_no_key(self, monkeypatch):
        monkeypatch.setenv("CHURN_SUBJECT_KEY_SALT", "a-real-secret")

        assert subject_key(None) is None
        assert subject_key("") is None
        assert subject_key("   ") is None


class TestUnicodeNormalisation:
    def test_nfc_and_nfd_forms_produce_the_same_key(self, monkeypatch):
        """
        The GDPR erasure bug: visually identical identifiers that hash differently
        make an erasure request match zero rows while reporting success.
        """
        monkeypatch.setenv("CHURN_SUBJECT_KEY_SALT", "a-real-secret")

        nfc = unicodedata.normalize("NFC", "José")
        nfd = unicodedata.normalize("NFD", "José")

        assert nfc != nfd, "test is meaningless if the two forms are identical"
        assert subject_key(nfc) == subject_key(nfd)

    def test_surrounding_whitespace_is_ignored(self, monkeypatch):
        monkeypatch.setenv("CHURN_SUBJECT_KEY_SALT", "a-real-secret")

        assert subject_key("  customer-1  ") == subject_key("customer-1")
