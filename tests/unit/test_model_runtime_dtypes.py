import sys
from types import SimpleNamespace

import pytest

from app.core.model_runtime_dtypes import (
    model_first_parameter_dtype_name,
    normalize_torch_dtype_name,
    torch_dtype_from_name,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (" ", None),
        ("bf16", "bfloat16"),
        ("torch.bfloat16", "bfloat16"),
        ("fp16", "float16"),
        ("half", "float16"),
        ("float", "float32"),
        ("torch.float32", "float32"),
    ],
)
def test_normalize_torch_dtype_name_accepts_known_aliases(
    value: str | None,
    expected: str | None,
) -> None:
    assert normalize_torch_dtype_name(value) == expected


def test_normalize_torch_dtype_name_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        normalize_torch_dtype_name("int8")


def test_torch_dtype_from_name_resolves_optional_torch_dtype(monkeypatch) -> None:
    bfloat16_dtype = object()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=bfloat16_dtype))

    assert torch_dtype_from_name("bf16") is bfloat16_dtype
    assert torch_dtype_from_name(None) is None


def test_torch_dtype_from_name_reports_missing_torch_dtype_attribute(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    with pytest.raises(ValueError, match="does not expose dtype"):
        torch_dtype_from_name("bfloat16")


def test_model_first_parameter_dtype_name_reads_direct_model_parameter() -> None:
    class FakeParameter:
        dtype = "torch.bfloat16"

    class FakeModel:
        def parameters(self):
            return iter([FakeParameter()])

    assert model_first_parameter_dtype_name(FakeModel()) == "bfloat16"


def test_model_first_parameter_dtype_name_reads_nested_model_parameter() -> None:
    class FakeParameter:
        dtype = "torch.float16"

    class FakeNestedModel:
        def parameters(self):
            return iter([FakeParameter()])

    assert model_first_parameter_dtype_name(SimpleNamespace(model=FakeNestedModel())) == "float16"


def test_model_first_parameter_dtype_name_returns_none_for_empty_or_untyped_model() -> None:
    class EmptyModel:
        def parameters(self):
            return iter(())

    class UntypedParameter:
        pass

    class UntypedModel:
        def parameters(self):
            return iter([UntypedParameter()])

    assert model_first_parameter_dtype_name(EmptyModel()) is None
    assert model_first_parameter_dtype_name(UntypedModel()) is None
