# type: ignore

import pytest
import dataclasses
import datetime
import uuid
import decimal
import re
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
    Literal,
    Pattern,
    ForwardRef,
    NewType,
)

from rsb.json.json_schema_builder import JsonSchemaBuilder
from rsb.json.unsuported_type_error import UnsupportedTypeError


# --- Test Fixtures and Helper Classes ---


class SimpleEnum(Enum):
    A = "apple"
    B = "banana"


class IntEnum(Enum):
    ONE = 1
    TWO = 2


class MixedEnum(Enum):
    A = "text"
    B = 123
    C = None


class NonPrimitiveEnum(Enum):
    DATE = datetime.date(2023, 1, 1)
    REGEX = re.compile(r"abc")


@dataclasses.dataclass
class SimpleDataclass:
    """A simple dataclass."""

    id: int
    name: str
    is_active: bool = True
    tags: Optional[List[str]] = None
    metadata: Dict[str, Any] = dataclasses.field(
        default_factory=dict, metadata={"description": "Extra data"}
    )


@dataclasses.dataclass
class NestedDataclass:
    """Contains another dataclass."""

    item: SimpleDataclass
    count: int


@dataclasses.dataclass
class RecursiveDataclass:
    """A recursive structure."""

    name: str
    parent: Optional["RecursiveDataclass"] = None


# Resolve forward reference for the test class itself
RecursiveDataclass.__annotations__["parent"] = Optional[RecursiveDataclass]


class GenericClass:
    """A generic class simulating a simple model."""

    def __init__(
        self,
        key: uuid.UUID,
        value: float,
        description: Optional[str] = "default description",
    ):
        self.key = key
        self.value = value
        self.description = description

    # Annotations are crucial for reflection
    key: uuid.UUID
    value: float
    description: Optional[str]


UserId = NewType("UserId", int)

# --- Test Cases ---


@pytest.mark.parametrize(
    "input_type, expected_schema",
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (type(None), {"type": "null"}),
        (Any, {}),
        (datetime.datetime, {"type": "string", "format": "date-time"}),
        (datetime.date, {"type": "string", "format": "date"}),
        (uuid.UUID, {"type": "string", "format": "uuid"}),
        (bytes, {"type": "string", "format": "byte"}),
        (decimal.Decimal, {"type": "string", "format": "decimal"}),
        # Test NewType - should resolve to underlying type
        (UserId, {"type": "integer"}),
    ],
    ids=[
        "str",
        "int",
        "float",
        "bool",
        "None",
        "Any",
        "datetime",
        "date",
        "uuid",
        "bytes",
        "decimal",
        "NewType",
    ],
)
def test_primitive_and_standard_types(input_type: Any, expected_schema: dict[str, Any]):
    builder = JsonSchemaBuilder(input_type)
    schema = builder.build()
    # Check the core schema part, ignore $schema and definitions for primitives
    core_schema = {
        k: v for k, v in schema.items() if k not in ["$schema", "definitions"]
    }
    assert core_schema == expected_schema


# --- Enum Tests ---


def test_simple_enum():
    builder = JsonSchemaBuilder(SimpleEnum)
    expected = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SimpleEnum",
        "type": "string",
        "enum": ["apple", "banana"],
    }
    assert builder.build() == expected


def test_int_enum():
    builder = JsonSchemaBuilder(IntEnum)
    expected = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "IntEnum",
        "type": "integer",
        "enum": [1, 2],
    }
    assert builder.build() == expected


def test_mixed_enum():
    builder = JsonSchemaBuilder(MixedEnum)
    expected = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "MixedEnum",
        "anyOf": [
            {"type": "integer", "enum": [123]},
            {"type": "null"},  # Simplified null representation
            {"type": "string", "enum": ["text"]},
        ],
    }
    # Order of anyOf might vary, compare sets
    generated = builder.build()
    assert generated["$schema"] == expected["$schema"]
    assert generated["title"] == expected["title"]
    assert "anyOf" in generated
    # Sort based on type for consistent comparison
    generated_anyof = sorted(generated["anyOf"], key=lambda x: x["type"])  # type: ignore
    expected_anyof = sorted(expected["anyOf"], key=lambda x: x["type"])  # type: ignore
    assert generated_anyof == expected_anyof


def test_non_primitive_enum():
    with pytest.warns(UserWarning, match="contains non-primitive value"):
        builder = JsonSchemaBuilder(NonPrimitiveEnum)
        expected = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "NonPrimitiveEnum",
            # Test expects anyOf even for a single string type resulting from non-primitives
            "anyOf": [
                {"type": "string", "enum": sorted(["2023-01-01", "re.compile('abc')"])},
            ],
        }
        # The actual string representation of the regex might differ slightly
        generated = builder.build()
        assert generated["$schema"] == expected["$schema"]
        assert generated["title"] == expected["title"]
        assert "anyOf" in generated, "Schema should use anyOf for non-primitive enums"
        assert len(generated["anyOf"]) == 1
        assert generated["anyOf"][0]["type"] == "string"
        # Sort generated enum for comparison
        generated_enum = sorted(generated["anyOf"][0].get("enum", []))
        assert generated_enum == expected["anyOf"][0]["enum"]  # type: ignore


# --- Typing Generics Tests ---


@pytest.mark.parametrize(
    "input_type, expected_items_schema",
    [
        (List[int], {"type": "integer"}),
        (
            List[SimpleEnum],
            {"title": "SimpleEnum", "type": "string", "enum": ["apple", "banana"]},
        ),
        (List, {}),  # List without type param -> any item
        (list, {}),  # Bare list type -> any item
    ],
    ids=["List[int]", "List[Enum]", "List", "list"],
)
def test_list_types(input_type: Any, expected_items_schema: dict[str, Any]):
    builder = JsonSchemaBuilder(input_type)
    expected = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": expected_items_schema,
    }
    assert builder.build() == expected


@pytest.mark.parametrize(
    "input_type, expected_additional_props_schema, should_warn",
    [
        (Dict[str, int], {"type": "integer"}, False),
        (Dict[str, Any], {}, False),
        (Dict, {}, False),  # Bare Dict -> any value
        (dict, {}, False),  # Bare dict -> any value
        (Dict[int, str], {"type": "string"}, True),  # Non-string keys
    ],
    ids=["Dict[str, int]", "Dict[str, Any]", "Dict", "dict", "Dict[int, str]"],
)
def test_dict_types(
    input_type: Any, expected_additional_props_schema: dict[str, Any], should_warn: bool
):
    builder = JsonSchemaBuilder(input_type)
    expected = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": expected_additional_props_schema,
    }
    if should_warn:
        with pytest.warns(UserWarning, match="JSON object keys must be strings"):
            assert builder.build() == expected
    else:
        assert builder.build() == expected


@pytest.mark.parametrize(
    "input_type, expected_schema",
    [
        (Optional[str], {"anyOf": [{"type": "string"}, {"type": "null"}]}),
        (Union[int, str], {"anyOf": [{"type": "integer"}, {"type": "string"}]}),
        (
            Union[int, None],
            {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        ),  # Same as Optional
        (
            Union[SimpleEnum, bool],
            {
                "anyOf": [
                    {"type": "boolean"},  # Sorted order
                    {
                        "title": "SimpleEnum",
                        "type": "string",
                        "enum": ["apple", "banana"],
                    },
                ]
            },
        ),
    ],
    ids=["Optional[str]", "Union[int, str]", "Union[int, None]", "Union[Enum, bool]"],
)
def test_union_types(input_type: Any, expected_schema: dict[str, Any]):
    builder = JsonSchemaBuilder(input_type)
    generated = builder.build()
    assert generated["$schema"] == "http://json-schema.org/draft-07/schema#"

    # Extract the core part for comparison (ignore $schema)
    core_generated = {k: v for k, v in generated.items() if k != "$schema"}

    # Sort anyOf lists for consistent comparison
    if "anyOf" in core_generated:
        core_generated["anyOf"].sort(key=lambda x: x.get("type", str(x)))  # type: ignore
    if "anyOf" in expected_schema:
        expected_schema["anyOf"].sort(key=lambda x: x.get("type", str(x)))  # type: ignore

    assert core_generated == expected_schema


@pytest.mark.parametrize(
    "input_type, expected_schema_part",
    [
        (
            Tuple[str, int],
            {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": [{"type": "string"}, {"type": "integer"}],
            },
        ),
        (Tuple[int, ...], {"type": "array", "items": {"type": "integer"}}),
        (Tuple, {"type": "array"}),  # Bare tuple -> any array
        (tuple, {"type": "array"}),  # Bare tuple type -> any array
    ],
    ids=["Tuple[str, int]", "Tuple[int, ...]", "Tuple", "tuple"],
)
def test_tuple_types(input_type: Any, expected_schema_part: dict[str, Any]):
    builder = JsonSchemaBuilder(input_type)
    generated = builder.build()
    assert generated["$schema"] == "http://json-schema.org/draft-07/schema#"
    # Check only the relevant parts, ignoring $schema
    for key, value in expected_schema_part.items():
        assert key in generated
        assert generated[key] == value


@pytest.mark.parametrize(
    "input_type, expected_schema",
    [
        (Literal["a"], {"const": "a", "type": "string"}),
        (Literal[123], {"const": 123, "type": "integer"}),
        (Literal[True], {"const": True, "type": "boolean"}),
        (Literal["a", "b"], {"enum": ["a", "b"], "type": "string"}),
        (Literal[1, 2, 3], {"enum": [1, 2, 3], "type": "integer"}),
        (
            Literal["a", 1],
            {"enum": [1, "a"]},
        ),  # Mixed types -> no top-level type, sorted
        (Literal[None], {"const": None, "type": "null"}),
        # Test expectation corrected: type should be string, None handled by enum
        (Literal["A", None], {"enum": [None, "A"], "type": "string"}),
    ],
    ids=[
        "Literal[str]",
        "Literal[int]",
        "Literal[bool]",
        "Literal[str, str]",
        "Literal[int, int]",
        "Literal[str, int]",
        "Literal[None]",
        "Literal[str, None]",
    ],
)
def test_literal_types(input_type: Any, expected_schema: dict[str, Any]):
    builder = JsonSchemaBuilder(input_type)
    generated = builder.build()
    assert generated["$schema"] == "http://json-schema.org/draft-07/schema#"
    core_schema = {k: v for k, v in generated.items() if k != "$schema"}
    # Sort enum lists for consistent comparison
    if "enum" in core_schema:
        # Convert to tuple for sorting mixed types if necessary
        core_schema["enum"] = sorted(
            core_schema["enum"],
            key=lambda x: (isinstance(x, type(None)), type(x).__name__, x),
        )
    if "enum" in expected_schema:
        expected_schema["enum"] = sorted(
            expected_schema["enum"],
            key=lambda x: (isinstance(x, type(None)), type(x).__name__, x),
        )

    assert core_schema == expected_schema


def test_pattern_type():
    builder = JsonSchemaBuilder(Pattern)
    expected = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "string",
        "format": "regex",
    }
    assert builder.build() == expected


# --- Complex Types (Dataclasses, Classes, $ref) ---


def test_simple_dataclass():
    builder = JsonSchemaBuilder(SimpleDataclass)
    schema = builder.build()

    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$ref"] == "#/definitions/SimpleDataclass"
    assert "definitions" in schema
    assert "SimpleDataclass" in schema["definitions"]

    defn = schema["definitions"]["SimpleDataclass"]
    assert defn["title"] == "SimpleDataclass"
    assert defn["type"] == "object"
    assert defn["description"] == "A simple dataclass."
    assert "properties" in defn
    assert sorted(list(defn["properties"].keys())) == sorted(
        ["id", "name", "is_active", "tags", "metadata"]
    )  # Sort keys for comparison

    assert defn["properties"]["id"] == {"type": "integer"}
    assert defn["properties"]["name"] == {"type": "string"}
    assert defn["properties"]["is_active"] == {"type": "boolean", "default": True}
    assert defn["properties"]["tags"] == {
        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
        # Note: default_factory = list is not represented in standard JSON Schema default
    }
    assert defn["properties"]["metadata"] == {
        "type": "object",
        "additionalProperties": {},
        "description": "Extra data",
        # default_factory = dict not represented
    }

    assert defn["required"] == sorted(
        ["id", "name"]
    )  # is_active, tags, metadata have defaults/factories


def test_nested_dataclass():
    builder = JsonSchemaBuilder(NestedDataclass)
    schema = builder.build()

    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$ref"] == "#/definitions/NestedDataclass"
    assert "definitions" in schema
    assert "NestedDataclass" in schema["definitions"]
    assert (
        "SimpleDataclass" in schema["definitions"]
    )  # Nested type should also be defined

    defn = schema["definitions"]["NestedDataclass"]
    assert defn["title"] == "NestedDataclass"
    assert defn["description"] == "Contains another dataclass."
    assert defn["properties"]["item"] == {"$ref": "#/definitions/SimpleDataclass"}
    assert defn["properties"]["count"] == {"type": "integer"}
    # Corrected: Expect alphabetically sorted required list
    assert defn["required"] == ["count", "item"]


def test_recursive_dataclass():
    builder = JsonSchemaBuilder(RecursiveDataclass)
    schema = builder.build()

    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$ref"] == "#/definitions/RecursiveDataclass"
    assert "definitions" in schema
    assert "RecursiveDataclass" in schema["definitions"]

    defn = schema["definitions"]["RecursiveDataclass"]
    assert defn["title"] == "RecursiveDataclass"
    assert defn["description"] == "A recursive structure."
    assert "properties" in defn
    assert defn["properties"]["name"] == {"type": "string"}
    # Crucially, the recursive field should use $ref
    parent_prop = defn["properties"]["parent"]
    assert "anyOf" in parent_prop
    # Sort anyOf for comparison
    parent_prop["anyOf"].sort(key=lambda x: x.get("type", "$ref"))
    assert parent_prop["anyOf"] == [
        {"$ref": "#/definitions/RecursiveDataclass"},
        {"type": "null"},
    ]

    assert defn["required"] == ["name"]


def test_generic_class():
    builder = JsonSchemaBuilder(GenericClass)
    schema = builder.build()

    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$ref"] == "#/definitions/GenericClass"
    assert "definitions" in schema
    assert "GenericClass" in schema["definitions"]

    defn = schema["definitions"]["GenericClass"]
    assert defn["title"] == "GenericClass"
    assert defn["description"] == "A generic class simulating a simple model."
    assert "properties" in defn
    assert sorted(list(defn["properties"].keys())) == sorted(
        ["key", "value", "description"]
    )  # Sort keys

    assert defn["properties"]["key"] == {"type": "string", "format": "uuid"}
    assert defn["properties"]["value"] == {"type": "number"}
    # Sort anyOf for comparison
    desc_prop = defn["properties"]["description"]
    assert "anyOf" in desc_prop
    desc_prop["anyOf"].sort(key=lambda x: x["type"])
    assert desc_prop == {
        "anyOf": [
            {"type": "null"},
            {"type": "string"},
        ],
        "default": "default description",  # Default from __init__ captured
    }
    # 'key' and 'value' are required because they don't have defaults in __init__
    assert defn["required"] == sorted(["key", "value"])  # Sort required


# --- Builder Options ---


@dataclasses.dataclass
class ClassWithExamples:
    id: int = dataclasses.field(metadata={"examples": [101, 102]})
    name: str = "test"


def test_remove_examples_option():
    # Build WITH examples (default)
    builder_with = JsonSchemaBuilder(ClassWithExamples, remove_examples=False)
    schema_with = builder_with.build()
    assert (
        "examples"
        in schema_with["definitions"]["ClassWithExamples"]["properties"]["id"]
    )

    # Build WITHOUT examples
    builder_without = JsonSchemaBuilder(ClassWithExamples, remove_examples=True)
    schema_without = builder_without.build()
    assert (
        "examples"
        not in schema_without["definitions"]["ClassWithExamples"]["properties"]["id"]
    )


# Use importorskip for optional dependency jsonref
jsonref = pytest.importorskip("jsonref")


def test_dereference_option_true():
    builder = JsonSchemaBuilder(NestedDataclass)
    # Build without dereferencing (should have refs)
    schema_ref = builder.build(dereference=False)
    assert "$ref" in schema_ref
    assert "$ref" in schema_ref["definitions"]["NestedDataclass"]["properties"]["item"]

    # Build with dereferencing
    schema_deref = builder.build(dereference=True)

    # Check top level (if root was complex)
    assert "$ref" not in schema_deref  # Top level ref is resolved
    assert schema_deref["title"] == "NestedDataclass"

    # Check nested structure
    assert "$ref" not in schema_deref["properties"]["item"]  # Nested ref resolved
    assert schema_deref["properties"]["item"]["type"] == "object"
    assert schema_deref["properties"]["item"]["title"] == "SimpleDataclass"
    assert "id" in schema_deref["properties"]["item"]["properties"]

    # Definitions section might still exist if jsonref keeps it, but refs inside are resolved
    # Allow definitions section if present, but ensure it's empty or contains fully resolved schemas
    if "definitions" in schema_deref:
        assert not any(
            "$ref" in v
            for v in schema_deref["definitions"].values()
            if isinstance(v, dict)
        ), "Definitions should not contain $refs after dereferencing"


def test_dereference_option_true_without_jsonref(mocker: Any):
    # Não precisamos mais simular a ausência do jsonref, pois a implementação atual
    # usa dereferenciamento manual
    builder = JsonSchemaBuilder(NestedDataclass)

    # Apenas verifica se o dereferenciamento funciona como esperado
    schema_deref = builder.build(dereference=True)

    # Verifica se as referências foram resolvidas
    assert "$ref" not in schema_deref
    assert schema_deref["title"] == "NestedDataclass"
    assert "$ref" not in schema_deref["properties"]["item"]


# --- Error Handling ---


class UnknownType:
    pass


def test_unsupported_type_error():
    builder = JsonSchemaBuilder(UnknownType)
    with pytest.raises(UnsupportedTypeError, match="Cannot generate JSON Schema"):
        builder.build()


# Test ForwardRef resolution (basic case)
MyForwardRefClass = ForwardRef("MyForwardRefClass")  # type: ignore


@dataclasses.dataclass
class MyForwardRefClass:
    value: int
    next: Optional[MyForwardRefClass] = None


def test_forward_ref_resolution():
    builder = JsonSchemaBuilder(MyForwardRefClass)
    schema = builder.build()
    # If it builds without error, resolution worked. Check structure.
    assert schema["$ref"] == "#/definitions/MyForwardRefClass"
    defn = schema["definitions"]["MyForwardRefClass"]
    next_prop = defn["properties"]["next"]
    assert "anyOf" in next_prop
    # Sort anyOf for comparison
    next_prop["anyOf"].sort(key=lambda x: x.get("type", "$ref"))
    assert next_prop["anyOf"][0]["$ref"] == "#/definitions/MyForwardRefClass"


def test_unresolved_forward_ref():
    # Define a forward ref that won't be resolvable in the current scope
    UnresolvedRef = ForwardRef("ThisClassDoesNotExist")  # type: ignore
    builder = JsonSchemaBuilder(UnresolvedRef)  # type: ignore
    with pytest.raises(
        UnsupportedTypeError,
        match="Unresolved forward reference: 'ThisClassDoesNotExist'",
    ):
        builder.build()


# --- Testes de Modelos Pydantic ---
try:
    # Tentativa de importação para Pydantic v2
    from pydantic import BaseModel, Field, model_validator, field_validator
    from pydantic import constr, conlist, conint, confloat
    from pydantic import ConfigDict

    HAS_PYDANTIC_V2 = True
    HAS_PYDANTIC = True
except ImportError:
    try:
        # Tentativa para Pydantic v1
        from pydantic import BaseModel, Field, validator, root_validator
        from pydantic import constr, conlist, conint, confloat

        # discriminator não existe em v1 da mesma forma
        HAS_PYDANTIC_V2 = False
        HAS_PYDANTIC = True
    except ImportError as e:
        import warnings
        import traceback

        warnings.warn(
            f"Pydantic não está instalado ou ocorreu um erro na importação: {e}\n{traceback.format_exc()}"
        )
        HAS_PYDANTIC = False
        HAS_PYDANTIC_V2 = False


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic não está instalado")
class TestPydanticModels:
    """Testes para modelos Pydantic."""

    def test_basic_pydantic_model(self):
        """Teste básico de modelo Pydantic."""

        class User(BaseModel):
            id: int
            name: str
            email: Optional[str] = None
            is_active: bool = True

        builder = JsonSchemaBuilder(User)
        schema = builder.build()

        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["$ref"] == "#/definitions/User"
        assert "User" in schema["definitions"]

        defn = schema["definitions"]["User"]
        assert defn["title"] == "User"
        assert defn["type"] == "object"
        assert sorted(list(defn["properties"].keys())) == sorted(
            ["id", "name", "email", "is_active"]
        )

        assert defn["properties"]["id"] == {"type": "integer"}
        assert defn["properties"]["name"] == {"type": "string"}
        assert defn["properties"]["email"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }
        assert defn["properties"]["is_active"] == {"type": "boolean", "default": True}

        assert sorted(defn["required"]) == sorted(["id", "name"])

    def test_pydantic_field_constraints(self):
        """Teste de modelo Pydantic com restrições de Field."""

        class Product(BaseModel):
            id: int = Field(gt=0, description="ID único do produto")
            name: str = Field(
                min_length=3, max_length=50, description="Nome do produto"
            )
            price: float = Field(ge=0.01, le=1000000, description="Preço do produto")
            tags: List[str] = Field(
                min_length=1, max_length=10, description="Tags do produto"
            )
            stock: int = Field(ge=0, default=0, description="Quantidade em estoque")

        builder = JsonSchemaBuilder(Product)
        schema = builder.build()

        defn = schema["definitions"]["Product"]

        assert defn["properties"]["id"]["type"] == "integer"
        assert defn["properties"]["id"]["description"] == "ID único do produto"
        assert defn["properties"]["id"]["exclusiveMinimum"] == 0

        assert defn["properties"]["name"]["type"] == "string"
        assert defn["properties"]["name"]["minLength"] == 3
        assert defn["properties"]["name"]["maxLength"] == 50

        assert defn["properties"]["price"]["type"] == "number"
        assert defn["properties"]["price"]["minimum"] == 0.01
        assert defn["properties"]["price"]["maximum"] == 1000000

        assert defn["properties"]["tags"]["type"] == "array"
        assert defn["properties"]["tags"]["minItems"] == 1
        assert defn["properties"]["tags"]["maxItems"] == 10

        assert defn["properties"]["stock"]["default"] == 0

    def test_pydantic_nested_models(self):
        """Teste de modelos Pydantic aninhados."""

        class Address(BaseModel):
            street: str
            city: str
            country: str
            postal_code: str

        class Contact(BaseModel):
            email: str
            phone: Optional[str] = None

        class Customer(BaseModel):
            id: int
            name: str
            address: Address
            contacts: List[Contact]
            metadata: Dict[str, Any] = {}

        builder = JsonSchemaBuilder(Customer)
        schema = builder.build()

        assert "Customer" in schema["definitions"]
        assert "Address" in schema["definitions"]
        assert "Contact" in schema["definitions"]

        customer_def = schema["definitions"]["Customer"]
        assert customer_def["properties"]["address"]["$ref"] == "#/definitions/Address"
        assert customer_def["properties"]["contacts"]["type"] == "array"
        assert (
            customer_def["properties"]["contacts"]["items"]["$ref"]
            == "#/definitions/Contact"
        )

    def test_pydantic_with_validators(self):
        """Teste de modelo Pydantic com validadores."""
        if HAS_PYDANTIC_V2:

            class User(BaseModel):
                username: str = Field(min_length=3, max_length=20)
                password: str = Field(min_length=8)
                password_confirm: str

                @field_validator("password_confirm")
                def passwords_match(cls, v, info):
                    if "password" in info.data and v != info.data["password"]:
                        raise ValueError("Passwords do not match")
                    return v

                @model_validator(mode="after")
                def check_fields(self):
                    return self
        else:

            class User(BaseModel):
                username: str = Field(min_length=3, max_length=20)
                password: str = Field(min_length=8)
                password_confirm: str

                @validator("password_confirm")
                def passwords_match(cls, v, values):
                    if "password" in values and v != values["password"]:
                        raise ValueError("Passwords do not match")
                    return v

                @root_validator
                def check_card_number_omitted(cls, values):
                    return values

        builder = JsonSchemaBuilder(User)
        schema = builder.build()

        defn = schema["definitions"]["User"]
        assert defn["required"] == sorted(["password", "password_confirm", "username"])
        assert defn["properties"]["username"]["minLength"] == 3
        assert defn["properties"]["username"]["maxLength"] == 20
        assert defn["properties"]["password"]["minLength"] == 8

    def test_pydantic_with_config(self):
        """Teste de modelo Pydantic com configurações personalizadas."""

        class UserModel(BaseModel):
            model_config = ConfigDict(title="UserSchema", extra="forbid")

            id: int
            name: str
            metadata: Dict[str, Any] = Field(default_factory=dict)

        builder = JsonSchemaBuilder(UserModel)
        schema = builder.build()

        defn = schema["definitions"]["UserModel"]
        assert defn["title"] == "UserSchema"  # Título da configuração
        assert defn.get(
            "additionalProperties"
        )  # extra="forbid" deve proibir propriedades adicionais

    def test_pydantic_constrained_types(self):
        """Teste de tipos restritos do Pydantic."""

        class ModelWithConstrainedTypes(BaseModel):
            text: constr(min_length=5, max_length=100, pattern=r"^[a-zA-Z0-9]+$")
            numbers: conlist(conint(ge=0, le=100), min_length=1, max_length=10)
            ratio: confloat(ge=0.0, le=1.0)

        builder = JsonSchemaBuilder(ModelWithConstrainedTypes)
        schema = builder.build()

        defn = schema["definitions"]["ModelWithConstrainedTypes"]

        # Verificar constr
        assert defn["properties"]["text"]["type"] == "string"
        assert defn["properties"]["text"]["minLength"] == 5
        assert defn["properties"]["text"]["maxLength"] == 100
        assert defn["properties"]["text"]["pattern"] == r"^[a-zA-Z0-9]+$"

        # Verificar conlist com conint
        assert defn["properties"]["numbers"]["type"] == "array"
        assert defn["properties"]["numbers"]["minItems"] == 1
        assert defn["properties"]["numbers"]["maxItems"] == 10
        assert defn["properties"]["numbers"]["items"]["type"] == "integer"
        assert defn["properties"]["numbers"]["items"]["minimum"] == 0
        assert defn["properties"]["numbers"]["items"]["maximum"] == 100

        # Verificar confloat
        assert defn["properties"]["ratio"]["type"] == "number"
        assert defn["properties"]["ratio"]["minimum"] == 0.0
        assert defn["properties"]["ratio"]["maximum"] == 1.0

    @pytest.mark.skipif(
        not hasattr(BaseModel, "model_discriminator"),
        reason="Discriminator não suportado nesta versão de Pydantic",
    )
    def test_pydantic_discriminated_unions(self):
        """Teste de uniões discriminadas do Pydantic."""
        if HAS_PYDANTIC_V2:

            class Animal(BaseModel):
                name: str

            class Dog(Animal):
                bark: str

            class Cat(Animal):
                meow: str

            class Pet(BaseModel):
                pet: Union[Dog, Cat] = Field(discriminator="pet_type")
                owner: str
        else:
            # Pular em versões antigas
            pytest.skip(
                "Discriminated unions não são suportados nesta versão do Pydantic"
            )

        builder = JsonSchemaBuilder(Pet)
        schema = builder.build()

        assert "Dog" in schema["definitions"]
        assert "Cat" in schema["definitions"]
        assert "Animal" in schema["definitions"]
        assert "Pet" in schema["definitions"]

        pet_schema = schema["definitions"]["Pet"]["properties"]["pet"]
        assert "discriminator" in pet_schema
        assert pet_schema["discriminator"]["propertyName"] == "pet_type"
        assert "oneOf" in pet_schema
        assert len(pet_schema["oneOf"]) == 2

    def test_pydantic_literal_types(self):
        """Teste de tipos Literal em modelos Pydantic."""

        class Status(BaseModel):
            state: Literal["active", "inactive", "pending"]
            priority: Literal[1, 2, 3] = 1
            flag: Optional[Literal[True]] = None
            mixed: Union[Literal["high", "low"], None, int] = "low"

        builder = JsonSchemaBuilder(Status)
        schema = builder.build()

        defn = schema["definitions"]["Status"]

        # Verificar Literal de strings
        assert defn["properties"]["state"]["enum"] == ["active", "inactive", "pending"]
        assert defn["properties"]["state"]["type"] == "string"

        # Verificar Literal de inteiros com valor padrão
        assert defn["properties"]["priority"]["enum"] == [1, 2, 3]
        assert defn["properties"]["priority"]["type"] == "integer"
        assert defn["properties"]["priority"]["default"] == 1

        # Verificar Optional[Literal[bool]]
        assert "anyOf" in defn["properties"]["flag"]
        flag_types = [
            item.get("type")
            for item in defn["properties"]["flag"]["anyOf"]
            if "type" in item
        ]
        assert "null" in flag_types
        assert any(
            item.get("const")
            for item in defn["properties"]["flag"]["anyOf"]
            if "const" in item
        )

        # Verificar Union complexo com Literal
        assert "anyOf" in defn["properties"]["mixed"]
        mixed_schemas = defn["properties"]["mixed"]["anyOf"]
        has_literal_high_low = any(
            item.get("enum") == ["high", "low"]
            for item in mixed_schemas
            if "enum" in item
        )
        has_null = any(item.get("type") == "null" for item in mixed_schemas)
        has_integer = any(item.get("type") == "integer" for item in mixed_schemas)
        assert has_literal_high_low and has_null and has_integer

    def test_pydantic_complex_inheritance(self):
        """Teste de herança complexa em modelos Pydantic."""

        class Base(BaseModel):
            id: int
            created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

        class Mixin:
            updated_at: Optional[datetime.datetime] = None

        class Derived(Base, Mixin):
            name: str
            data: Dict[str, Any] = {}

        builder = JsonSchemaBuilder(Derived)
        schema = builder.build()

        assert "Derived" in schema["definitions"]
        defn = schema["definitions"]["Derived"]

        # Deve herdar campos da classe Base
        assert "id" in defn["properties"]
        assert "created_at" in defn["properties"]

        # Deve incluir campos do Mixin
        assert "updated_at" in defn["properties"]

        # Deve incluir seus próprios campos
        assert "name" in defn["properties"]
        assert "data" in defn["properties"]

        # Verificar campos required
        assert "id" in defn["required"]
        assert "name" in defn["required"]

    @pytest.mark.skipif(
        not hasattr(BaseModel, "model_config"),
        reason="model_config não disponível nesta versão de Pydantic",
    )
    def test_pydantic_v2_specific_features(self):
        """Teste de recursos específicos do Pydantic v2."""

        class User(BaseModel):
            model_config = ConfigDict(
                json_schema_extra={
                    "examples": [
                        {"id": 123, "name": "John Doe", "email": "john@example.com"}
                    ]
                }
            )

            id: int
            name: str
            email: str

        builder = JsonSchemaBuilder(User)
        schema = builder.build()

        defn = schema["definitions"]["User"]
        assert "examples" in defn
        assert len(defn["examples"]) == 1
        assert defn["examples"][0]["id"] == 123


# --- Testes de casos extremamente complexos ---


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic não está instalado")
def test_extremely_complex_nested_model():
    """Teste com um modelo extremamente complexo com muitos níveis de aninhamento."""

    class GeoPoint(BaseModel):
        lat: float = Field(ge=-90, le=90)
        lng: float = Field(ge=-180, le=180)

    class Address(BaseModel):
        street: str
        number: Optional[int] = None
        city: str
        country: str
        geo: Optional[GeoPoint] = None

    class ContactInfo(BaseModel):
        emails: List[str] = Field(min_length=1)
        phones: Dict[str, str] = {}  # tipo -> número

    class Role(str, Enum):
        ADMIN = "admin"
        USER = "user"
        GUEST = "guest"

    class Permission(BaseModel):
        resource: str
        actions: List[Literal["read", "write", "delete"]]

    class UserSettings(BaseModel):
        theme: Literal["light", "dark", "system"] = "system"
        notifications: bool = True
        language: str = "en"

    class UserProfile(BaseModel):
        bio: Optional[str] = None
        avatar_url: Optional[str] = None
        links: Dict[str, str] = {}  # plataforma -> url

    class User(BaseModel):
        id: uuid.UUID
        username: str = Field(min_length=3, max_length=20)
        password_hash: str
        role: Role = Role.USER
        permissions: List[Permission] = []
        contact: ContactInfo
        addresses: Dict[Literal["home", "work", "other"], Address] = {}
        settings: UserSettings = UserSettings()
        profile: Optional[UserProfile] = None
        is_active: bool = True
        created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
        last_login: Optional[datetime.datetime] = None
        metadata: Dict[str, Any] = {}

    class Comment(BaseModel):
        id: int
        content: str
        author: User
        created_at: datetime.datetime
        parent: Optional["Comment"] = None

    Comment.model_rebuild()  # Resolver referência forward

    class Post(BaseModel):
        id: int
        title: str
        content: str
        author: User
        comments: List[Comment] = []
        tags: List[str] = []
        published: bool = False
        views: int = 0
        created_at: datetime.datetime
        updated_at: Optional[datetime.datetime] = None

    builder = JsonSchemaBuilder(Post)
    schema = builder.build()

    # Verificações básicas
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$ref"] == "#/definitions/Post"

    # Verificar se todas as definições existem
    expected_definitions = [
        "Post",
        "User",
        "Comment",
        "Address",
        "GeoPoint",
        "ContactInfo",
        "Permission",
        "UserSettings",
        "UserProfile",
    ]
    for defn in expected_definitions:
        assert defn in schema["definitions"], f"Definição de {defn} não encontrada"

    # Verificar alguns elementos de aninhamento profundo
    post_def = schema["definitions"]["Post"]
    assert post_def["properties"]["author"]["$ref"] == "#/definitions/User"
    assert post_def["properties"]["comments"]["type"] == "array"
    assert (
        post_def["properties"]["comments"]["items"]["$ref"] == "#/definitions/Comment"
    )

    user_def = schema["definitions"]["User"]
    assert user_def["properties"]["role"]["enum"] == ["admin", "user", "guest"]
    assert user_def["properties"]["contact"]["$ref"] == "#/definitions/ContactInfo"

    # Verificar referência recursiva
    comment_def = schema["definitions"]["Comment"]
    assert "anyOf" in comment_def["properties"]["parent"]
    parent_refs = [
        item.get("$ref")
        for item in comment_def["properties"]["parent"]["anyOf"]
        if "$ref" in item
    ]
    assert "#/definitions/Comment" in parent_refs


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic não está instalado")
def test_complex_recursive_model():
    """Teste com modelo recursivo complexo."""

    # Definição da classe Node com referências recursivas
    class Node(BaseModel):
        id: int
        name: str
        children: List["Node"] = []
        parent: Optional["Node"] = None
        data: Dict[str, Any] = {}
        metadata: Optional[Dict[str, Any]] = None

    # Resolve as referências forward
    Node.model_rebuild()

    # Construa o schema
    builder = JsonSchemaBuilder(Node)
    schema = builder.build()

    # Verificar se temos a definição correta
    assert "Node" in schema["definitions"]
    node_def = schema["definitions"]["Node"]

    # Verificar estrutura recursiva para 'children'
    assert node_def["properties"]["children"]["type"] == "array"
    assert node_def["properties"]["children"]["items"]["$ref"] == "#/definitions/Node"

    # Verificar estrutura para Optional[Node]
    assert "anyOf" in node_def["properties"]["parent"]
    assert len(node_def["properties"]["parent"]["anyOf"]) == 2
    assert {"$ref": "#/definitions/Node"} in node_def["properties"]["parent"]["anyOf"]
    assert {"type": "null"} in node_def["properties"]["parent"]["anyOf"]


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic não está instalado")
def test_union_with_multiple_complex_types():
    """Teste de união com múltiplos tipos complexos."""

    class ImageData(BaseModel):
        url: str
        width: int
        height: int

    class VideoData(BaseModel):
        url: str
        duration: float
        format: str

    class DocumentData(BaseModel):
        url: str
        pages: int
        file_size: int

    class Media(BaseModel):
        id: int
        title: str
        content: Union[ImageData, VideoData, DocumentData, Literal["placeholder"], None]

    builder = JsonSchemaBuilder(Media)
    schema = builder.build()

    assert "Media" in schema["definitions"]
    assert "ImageData" in schema["definitions"]
    assert "VideoData" in schema["definitions"]
    assert "DocumentData" in schema["definitions"]

    media_def = schema["definitions"]["Media"]
    content_prop = media_def["properties"]["content"]

    assert "anyOf" in content_prop

    # Verificar se todas as opções possíveis estão presentes
    refs = [item.get("$ref") for item in content_prop["anyOf"] if "$ref" in item]
    assert "#/definitions/ImageData" in refs
    assert "#/definitions/VideoData" in refs
    assert "#/definitions/DocumentData" in refs

    # Verificar Literal e None
    has_literal = any(
        item.get("enum") == ["placeholder"]
        for item in content_prop["anyOf"]
        if "enum" in item and "type" in item and item.get("type") == "string"
    )
    has_null = any(item.get("type") == "null" for item in content_prop["anyOf"])

    assert has_literal
    assert has_null
