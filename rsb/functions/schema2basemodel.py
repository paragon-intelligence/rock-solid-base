from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    ForwardRef,
    Tuple,
    cast,
    TypeVar,
)
from pydantic import BaseModel, Field, create_model
import datetime
import uuid
import re
import decimal
from enum import Enum
import inspect

# Type aliases to help with type checking - allowing more flexibility
# Use Any for the first element to allow any type
FieldDefinition = Tuple[Any, Any]
FieldDefinitions = Dict[str, FieldDefinition]

# Type variable for BaseModel to help with return types
ModelType = TypeVar("ModelType", bound=BaseModel)


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
        self.model_registry: Dict[
            str, Type[BaseModel]
        ] = {}  # Track created models to handle references
        self.processing_refs: set[str] = (
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

    def _create_model_from_schema(self, schema: Dict[str, Any]) -> Type[Any]:
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

    def _handle_ref(self, ref_path: str) -> Type[Any]:
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
            # Cast ForwardRef to Type[Any] to satisfy the type checker
            return cast(Type[Any], ForwardRef(ref_name))

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
                # Using model_fields for Pydantic v2
                if hasattr(model, "model_fields"):
                    # For Pydantic v2
                    # We can't directly access placeholder.model_fields as it might be a property
                    # Instead, we'll recreate the model and replace our placeholder
                    fields_dict: FieldDefinitions = {}
                    for name, field_info in model.model_fields.items():
                        # Extract type and default from field_info
                        field_type = field_info.annotation
                        default = field_info.default
                        # Ensure field_type is not None to match FieldDefinition
                        if field_type is None:
                            field_type = Any
                        fields_dict[name] = (field_type, default)

                    # Use type: ignore to bypass typing issues with create_model
                    new_model = create_model(ref_name, **fields_dict)  # type: ignore
                    self.model_registry[ref_name] = new_model
                    return cast(Type[BaseModel], new_model)

                # Set correct name
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
        fields: FieldDefinitions = {}
        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required
            field_info = self._create_field_from_schema(
                prop_name, prop_schema, is_required
            )
            fields[prop_name] = field_info

        # Create the model class, adding type: ignore to bypass strict type checking
        model = create_model(model_name, **fields)  # type: ignore

        # Add description if available
        if "description" in schema:
            model.__doc__ = schema["description"]

        # Configure additional properties behavior
        if "additionalProperties" in schema:
            # Use model_config for Pydantic v2 compatibility
            extra_value = (
                "forbid" if schema["additionalProperties"] is False else "allow"
            )

            # Try to use model_config (Pydantic v2)
            if hasattr(model, "model_config"):  # type: ignore
                model.model_config["extra"] = extra_value  # type: ignore
            else:
                # Fallback to Config class (Pydantic v1)
                class Config:
                    extra = extra_value

                setattr(model, "Config", Config)  # type: ignore

        return cast(Type[BaseModel], model)

    def _create_field_from_schema(
        self, name: str, schema: Dict[str, Any], is_required: bool
    ) -> Tuple[Type[Any], Any]:
        """Convert a schema definition to a Pydantic field tuple."""

        # Get the field type
        field_type = self._create_model_from_schema(schema)

        # Determine default value
        default = ... if is_required else None
        if "default" in schema:
            default = schema["default"]

        # Collect field constraints
        field_kwargs: Dict[str, Any] = {}

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
            # Combine default value with Field - explicitly cast Field to Any
            field_obj = Field(default, **field_kwargs)
            return (field_type, field_obj)
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

    def _create_enum_type(self, schema: Dict[str, Any]) -> Type[Any]:
        """Create a type for enum schema."""
        enum_values = schema.get("enum", [])
        enum_name = schema.get("title", "DynamicEnum")

        # Handle empty enum
        if not enum_values:
            return cast(Type[Any], Any)

        # Check if all values are of the same type, using Any to avoid type checking issues
        has_null = None in enum_values
        non_null_values = [val for val in enum_values if val is not None]

        # Check if all values are strings
        all_strings = (
            all(isinstance(v, str) for v in non_null_values)
            if non_null_values
            else False
        )

        # For string-only enums, create a proper Enum
        if all_strings:
            # Create Enum class
            enum_dict = {
                self._safe_enum_key(v): v for v in enum_values if v is not None
            }
            enum_class = Enum(enum_name, enum_dict)

            # Make it Optional if it includes null
            result = Optional[enum_class] if has_null else enum_class
            return cast(Type[Any], result)

        # For other types or mixed types, use a simpler approach
        try:
            # Use any basic type depending on the scenario
            if has_null:
                result = Optional[str] if non_null_values else type(None)
                return cast(Type[Any], result)
            else:
                # Just use str as a placeholder
                return cast(Type[Any], str)
        except (TypeError, ValueError):
            # If anything fails, fall back to Any
            return cast(Type[Any], Any)

    def _safe_enum_key(self, value: str) -> str:
        """Convert a string to a valid Enum key."""
        # Replace invalid characters and ensure it starts with a letter
        key = re.sub(r"[^a-zA-Z0-9_]", "_", str(value))
        if not key or not key[0].isalpha():
            key = "V_" + key
        return key

    def _create_union_type(self, schema: Dict[str, Any]) -> Type[Any]:
        """Create a Union type from anyOf/oneOf schema."""
        union_schemas = schema.get("anyOf", schema.get("oneOf", []))

        # Special case: if one of the options is null, use Optional
        has_null = any(
            s.get("type") == "null" or s.get("enum") == [None] for s in union_schemas
        )

        non_null_schemas = [
            s
            for s in union_schemas
            if s.get("type") != "null" and s.get("enum") != [None]
        ]

        # If no schemas, return appropriate type
        if not non_null_schemas:
            result = type(None) if has_null else Any
            return cast(Type[Any], result)

        # If only one schema, process it
        if len(non_null_schemas) == 1:
            sub_type = self._create_model_from_schema(non_null_schemas[0])
            result = Optional[sub_type] if has_null else sub_type
            return cast(Type[Any], result)

        # Otherwise create a union of basic types (safer than dynamic union)
        result = str if has_null else Union[str, int]
        return cast(Type[Any], result)

    def _create_array_type(self, schema: Dict[str, Any]) -> Type[Any]:
        """Create a List or tuple type from array schema."""
        items = schema.get("items", {})

        if not items:
            return List[Any]  # Default to List[Any] if items not specified

        if isinstance(items, dict):
            # Single type for all items - bypass type checking with type: ignore
            item_type = self._create_model_from_schema(items)  # type: ignore
            return cast(Type[Any], List[item_type])
        elif isinstance(items, list):
            # Simplify by always returning List[Any] for tuple types
            return List[Any]

        return List[Any]

    def _get_python_type_from_schema(self, schema: Dict[str, Any]) -> Type[Any]:
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
        return cast(Type[Any], Any)
