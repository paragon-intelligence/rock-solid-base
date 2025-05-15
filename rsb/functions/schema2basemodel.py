from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    ForwardRef,
    Tuple,
    get_origin,
    get_args,
    Literal,
)
from pydantic import BaseModel, Field, create_model, validator
import datetime
import uuid
import re
import decimal
from enum import Enum
import inspect


def schema2basemodel(schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Convert a JSON Schema to a Pydantic BaseModel.

    Args:
        schema: A JSON Schema dictionary representing a model

    Returns:
        A dynamically created Pydantic BaseModel class
    """
    converter = SchemaConverter(schema.get("definitions", {}))
    return converter.convert(schema)


class SchemaConverter:
    """Class to manage the conversion of JSON Schema to Pydantic models."""

    def __init__(self, definitions: Dict[str, Any]):
        self.definitions = definitions
        self.model_registry = {}  # Track created models to handle references
        self.processing_refs = (
            set()
        )  # Track references being processed to detect circular references

    def convert(self, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Convert a JSON Schema to a Pydantic model."""
        model = self._create_model_from_schema(schema)

        # Ensure we return a BaseModel
        if not (inspect.isclass(model) and issubclass(model, BaseModel)):
            # If we got a primitive type, wrap it
            model_name = schema.get("title", "DynamicModel")
            wrapper_model = create_model(
                model_name,
                value=(model, ...),  # Required field
            )
            return wrapper_model

        return model

    def _create_model_from_schema(self, schema: Dict[str, Any]) -> Type:
        """Create a Pydantic model or type from a schema."""

        # Handle $ref first
        if "$ref" in schema:
            return self._handle_ref(schema["$ref"])

        # Handle different schema types
        schema_type = schema.get("type")

        # Handle object type (complex type with properties)
        if schema_type == "object" or ("properties" in schema and "type" not in schema):
            return self._create_object_model(schema)

        # Handle enum type
        elif "enum" in schema:
            return self._create_enum_type(schema)

        # Handle anyOf/oneOf (Union types in Pydantic)
        elif "anyOf" in schema or "oneOf" in schema:
            return self._create_union_type(schema)

        # Handle array type
        elif schema_type == "array":
            return self._create_array_type(schema)

        # For primitive types, return the appropriate Python type
        return self._get_python_type_from_schema(schema)

    def _handle_ref(self, ref_path: str) -> Type:
        """Handle references in the schema."""
        if not ref_path.startswith("#/definitions/"):
            raise ValueError(
                f"Only refs starting with #/definitions/ are supported, got {ref_path}"
            )

        ref_name = ref_path.split("/")[-1]

        # Use cached model if available
        if ref_name in self.model_registry:
            return self.model_registry[ref_name]

        # Check if reference exists
        if ref_name not in self.definitions:
            raise ValueError(f"Reference not found: {ref_path}")

        # Handle circular references
        if ref_name in self.processing_refs:
            # Create a forward reference for circular dependencies
            return ForwardRef(ref_name)

        # Mark as being processed
        self.processing_refs.add(ref_name)

        try:
            # Create the actual model
            definition = self.definitions[ref_name]

            # Create a placeholder first to handle recursive references
            placeholder = create_model(ref_name)
            self.model_registry[ref_name] = placeholder

            # Create the real model
            model = self._create_model_from_schema(definition)

            # If the result is a BaseModel, update our placeholder
            if inspect.isclass(model) and issubclass(model, BaseModel):
                # Copy fields from the real model to the placeholder
                for name, field in model.__fields__.items():
                    placeholder.__fields__[name] = field

                # Copy Config if it exists
                if hasattr(model, "Config"):
                    placeholder.Config = model.Config

                # Ensure correct name
                placeholder.__name__ = ref_name

                return placeholder
            else:
                # For non-BaseModel types (like primitives), create a wrapper
                wrapper = create_model(
                    ref_name,
                    value=(model, ...),
                )
                self.model_registry[ref_name] = wrapper
                return wrapper
        finally:
            # Always remove from processing set when done
            self.processing_refs.remove(ref_name)

    def _create_object_model(self, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic model from an object schema."""
        # Get model name
        model_name = schema.get("title", "DynamicModel")

        # Get properties and required fields
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Generate field definitions
        fields = {}
        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required
            field_info = self._create_field_from_schema(
                prop_name, prop_schema, is_required
            )
            fields[prop_name] = field_info

        # Create the model class
        model = create_model(model_name, **fields)

        # Add description if available
        if "description" in schema:
            model.__doc__ = schema["description"]

        # Configure additional properties behavior
        if "additionalProperties" in schema:

            class Config:
                extra = "forbid" if schema["additionalProperties"] is False else "allow"

            model.Config = Config

        return model

    def _create_field_from_schema(
        self, name: str, schema: Dict[str, Any], is_required: bool
    ) -> Tuple:
        """Convert a schema definition to a Pydantic field tuple."""

        # Get the field type
        field_type = self._create_model_from_schema(schema)

        # Determine default value
        default = ... if is_required else None
        if "default" in schema:
            default = schema["default"]

        # Collect field constraints
        field_kwargs = {}

        # Add metadata
        if "description" in schema:
            field_kwargs["description"] = schema["description"]

        if "title" in schema and schema["title"] != name:
            field_kwargs["title"] = schema["title"]

        if "examples" in schema:
            field_kwargs["examples"] = schema["examples"]

        # Map schema validations to Pydantic Field constraints
        self._add_validations_to_field(schema, field_kwargs)

        # Create the field - always return a two-tuple (type, value)
        if field_kwargs:
            # Combine default value with Field
            return (field_type, Field(default, **field_kwargs))
        else:
            return (field_type, default)

    def _add_validations_to_field(
        self, schema: Dict[str, Any], field_kwargs: Dict[str, Any]
    ):
        """Add JSON Schema validations to Pydantic Field kwargs."""
        schema_type = schema.get("type")

        # String validations
        if schema_type == "string":
            if "pattern" in schema:
                field_kwargs["regex"] = schema["pattern"]
            if "minLength" in schema:
                field_kwargs["min_length"] = schema["minLength"]
            if "maxLength" in schema:
                field_kwargs["max_length"] = schema["maxLength"]

        # Number validations
        elif schema_type in ["number", "integer"]:
            if "minimum" in schema:
                field_kwargs["ge"] = schema["minimum"]
            if "maximum" in schema:
                field_kwargs["le"] = schema["maximum"]
            if "exclusiveMinimum" in schema:
                field_kwargs["gt"] = schema["exclusiveMinimum"]
            if "exclusiveMaximum" in schema:
                field_kwargs["lt"] = schema["exclusiveMaximum"]
            if "multipleOf" in schema:
                field_kwargs["multiple_of"] = schema["multipleOf"]

        # Array validations
        elif schema_type == "array":
            if "minItems" in schema:
                field_kwargs["min_items"] = schema["minItems"]
            if "maxItems" in schema:
                field_kwargs["max_items"] = schema["maxItems"]

    def _create_enum_type(self, schema: Dict[str, Any]) -> Type:
        """Create a type for enum schema."""
        enum_values = schema.get("enum", [])
        enum_name = schema.get("title", "DynamicEnum")

        # Handle empty enum
        if not enum_values:
            return Any

        # Check if all values are of the same type
        types_present = set(type(val) for val in enum_values if val is not None)
        has_null = None in enum_values

        # For string-only enums, create a proper Enum
        if len(types_present) == 1 and list(types_present)[0] is str:
            # Create Enum class
            enum_dict = {
                self._safe_enum_key(v): v for v in enum_values if v is not None
            }
            enum_class = Enum(enum_name, enum_dict)

            # Make it Optional if it includes null
            return Optional[enum_class] if has_null else enum_class

        # For other types or mixed types, use Literal
        try:
            # Try to create a Literal type
            if has_null:
                non_null_values = tuple(v for v in enum_values if v is not None)
                return (
                    Optional[Literal[non_null_values]]
                    if non_null_values
                    else type(None)
                )
            else:
                return Literal[tuple(enum_values)]
        except (TypeError, ValueError):
            # If Literal creation fails (e.g., unhashable values), fall back to Any
            return Any

    def _safe_enum_key(self, value: str) -> str:
        """Convert a string to a valid Enum key."""
        # Replace invalid characters and ensure it starts with a letter
        key = re.sub(r"[^a-zA-Z0-9_]", "_", str(value))
        if not key or not key[0].isalpha():
            key = "V_" + key
        return key

    def _create_union_type(self, schema: Dict[str, Any]) -> Type:
        """Create a Union type from anyOf/oneOf schema."""
        union_schemas = schema.get("anyOf", schema.get("oneOf", []))
        union_types = []

        # Special case: if one of the options is null, use Optional
        has_null = any(
            s.get("type") == "null" or s.get("enum") == [None] for s in union_schemas
        )

        non_null_schemas = [
            s
            for s in union_schemas
            if s.get("type") != "null" and s.get("enum") != [None]
        ]

        # Process non-null schemas
        for sub_schema in non_null_schemas:
            sub_type = self._create_model_from_schema(sub_schema)
            union_types.append(sub_type)

        # Handle special cases
        if not union_types:
            return type(None) if has_null else Any
        elif len(union_types) == 1:
            return Optional[union_types[0]] if has_null else union_types[0]
        else:
            # Create Union type
            union_type = Union[tuple(union_types)]
            return Optional[union_type] if has_null else union_type

    def _create_array_type(self, schema: Dict[str, Any]) -> Type:
        """Create a List or tuple type from array schema."""
        items = schema.get("items", {})

        if not items:
            return List[Any]  # Default to List[Any] if items not specified

        if isinstance(items, dict):
            # Single type for all items
            item_type = self._create_model_from_schema(items)
            return List[item_type]
        elif isinstance(items, list):
            # Tuple with different types
            tuple_types = []
            for item_schema in items:
                item_type = self._create_model_from_schema(item_schema)
                tuple_types.append(item_type)

            if tuple_types:
                return Tuple[tuple(tuple_types)]
            else:
                return List[Any]

        return List[Any]

    def _get_python_type_from_schema(self, schema: Dict[str, Any]) -> Type:
        """Get the Python type corresponding to a JSON Schema type."""
        schema_type = schema.get("type")

        if schema_type == "string":
            if "format" in schema:
                fmt = schema["format"]
                if fmt == "date-time":
                    return datetime.datetime
                elif fmt == "date":
                    return datetime.date
                elif fmt == "time":
                    return datetime.time
                elif fmt == "uuid":
                    return uuid.UUID
                elif fmt == "byte":
                    return bytes
                elif fmt == "decimal":
                    return decimal.Decimal
            return str

        elif schema_type == "integer":
            return int

        elif schema_type == "number":
            return float

        elif schema_type == "boolean":
            return bool

        elif schema_type == "null":
            return type(None)

        # Default to Any for unknown types
        from typing import Any

        return Any


schema = {
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "address": {
            "type": "object",
            "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
            "required": ["street", "city"],
        },
    },
    "required": ["name", "age"],
}

PersonModel = schema2basemodel(schema)
# Now you can use it like any other Pydantic model
person = PersonModel(name="John Doe", age=30, tags=["developer"])
print(person)
