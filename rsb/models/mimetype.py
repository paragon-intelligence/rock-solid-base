from __future__ import annotations

import re
from typing import Any, ClassVar, Optional, Union

# Importações necessárias para Pydantic V2
from pydantic import BeforeValidator, GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from typing_extensions import Annotated


# Validador para MimeType que será usado nos campos do modelo
def _validate_mime_type(v: Union[str, _MimeType]) -> _MimeType:
    """Valida um valor como MimeType.

    Args:
        v: Valor a ser validado (string ou instância _MimeType)

    Returns:
        Instância _MimeType validada

    Raises:
        ValueError: Se o valor não é um MimeType válido
    """
    if isinstance(v, _MimeType):
        return v
    return _MimeType(v)


class _MimeTypeConstraints:
    """Constraints for MIME type validation."""

    pattern: str = (
        r"^[a-zA-Z0-9][a-zA-Z0-9!#$&^_-]{0,126}/[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]{0,126}$"
    )

    def __init__(self, pattern: Optional[str] = None) -> None:
        """Initialize the constraints.

        Args:
            pattern: Optional regex pattern to override the default MIME type pattern
        """
        if pattern is not None:
            self.pattern = pattern


class _MimeType:
    """Type for MIME type strings, usable as Pydantic annotation."""

    _constraints: ClassVar[_MimeTypeConstraints] = _MimeTypeConstraints()
    _mime_type: str

    def __init__(self, mime_type: Union[str, _MimeType]) -> None:
        """Initializes and validates the MimeType."""
        if isinstance(mime_type, _MimeType):
            self._mime_type = mime_type._mime_type
        else:
            self._validate_and_set(mime_type)

    def _validate_and_set(self, mime_type: str) -> None:
        """Internal validation logic."""
        if not re.match(self._constraints.pattern, mime_type):
            # Erro de formato
            raise ValueError(f"Invalid MIME type format: '{mime_type}'")

        self._mime_type = mime_type

    @property
    def type(self) -> str:
        """Get the main type component of the MIME type.

        Returns:
            The main type (part before the slash)
        """
        return self._mime_type.split("/", 1)[0]

    @property
    def subtype(self) -> str:
        """Get the subtype component of the MIME type.

        Returns:
            The subtype (part after the slash)
        """
        return self._mime_type.split("/", 1)[1]

    def __str__(self) -> str:
        """Return the string representation of the MIME type."""
        return self._mime_type

    def __repr__(self) -> str:
        """Return the representative string of the MIME type."""
        return f"{self.__class__.__name__}({self._mime_type!r})"

    def __eq__(self, other: Any) -> bool:
        """Compare this MIME type with another for equality."""
        if isinstance(other, _MimeType):
            return self._mime_type == other._mime_type
        if isinstance(other, str):
            return self._mime_type == other
        return False

    def __hash__(self) -> int:
        """Return the hash of the MIME type."""
        return hash(self._mime_type)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Provides the Pydantic Core Schema for validation and serialization."""

        # Função de validação que Pydantic chamará.
        def validate_mimetype(value: Union[str, _MimeType]) -> _MimeType:
            try:
                if isinstance(value, _MimeType):
                    return value

                # A validação real ocorre ao instanciar a classe.
                return cls(value)
            except ValueError as e:
                raise PydanticCustomError(
                    "value_error",  # Tipo do erro (pode ser customizado)
                    "Value error: {error_message}",  # Template da mensagem
                    {"error_message": str(e)},  # Contexto com a mensagem dinâmica
                )

        # Esquema de união que aceita string ou _MimeType
        return core_schema.union_schema(
            [
                # Opção 1: Valida strings
                core_schema.chain_schema(
                    [
                        # Aceita qualquer string, a validação será feita pela função validate_mimetype
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(validate_mimetype),
                    ]
                ),
                # Opção 2: Valida instâncias _MimeType diretamente
                core_schema.is_instance_schema(
                    cls,
                    serialization=core_schema.plain_serializer_function_ser_schema(str),
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


# Definindo MimeType como o tipo validado que será usado diretamente nas anotações
MimeType = Annotated[Union[str, _MimeType], BeforeValidator(_validate_mime_type)]
