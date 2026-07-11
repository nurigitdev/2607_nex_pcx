import pytest

from app.core.embedding_provider_contract_sample_sets import (
    EmbeddingProviderContractSampleSetInput,
    InvalidEmbeddingProviderContractSampleSetError,
    validate_embedding_provider_contract_sample_set_input,
)


def test_validate_contract_sample_set_input_normalizes_values() -> None:
    validated = validate_embedding_provider_contract_sample_set_input(
        EmbeddingProviderContractSampleSetInput(
            sample_set_name=" default_samples ",
            description=" Contract smoke samples ",
            input_type="document",
            sample_texts=(" sample one ", "sample two"),
            is_default=True,
        )
    )

    assert validated.sample_set_name == "default_samples"
    assert validated.description == "Contract smoke samples"
    assert validated.sample_texts == ("sample one", "sample two")
    assert validated.is_default is True


@pytest.mark.parametrize(
    ("sample_input", "message"),
    [
        (
            EmbeddingProviderContractSampleSetInput(sample_set_name=" "),
            "sample_set_name",
        ),
        (
            EmbeddingProviderContractSampleSetInput(
                sample_set_name="bad_input_type",
                input_type="image",
            ),
            "Unsupported input_type",
        ),
        (
            EmbeddingProviderContractSampleSetInput(
                sample_set_name="empty_samples",
                sample_texts=(),
            ),
            "sample_texts",
        ),
        (
            EmbeddingProviderContractSampleSetInput(
                sample_set_name="blank_sample",
                sample_texts=(" ",),
            ),
            "sample_text",
        ),
    ],
)
def test_validate_contract_sample_set_input_rejects_invalid_values(
    sample_input: EmbeddingProviderContractSampleSetInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderContractSampleSetError, match=message):
        validate_embedding_provider_contract_sample_set_input(sample_input)
