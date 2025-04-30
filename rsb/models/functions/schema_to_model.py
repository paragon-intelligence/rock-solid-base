from typing import Any, Optional, Union, cast

from rsb.models.base_model import BaseModel
from rsb.models.create_model import create_model
from rsb.models.field import Field


def schema_to_model(
    schema: dict[str, Any], model_name: str = "Model"
) -> type[BaseModel]:
    """
    Dynamically creates a Pydantic BaseModel class from a JSON schema dictionary.

    Handles nested objects, arrays, required fields, unions (including nullable types),
    and descriptions.

    Args:
        schema: The JSON schema dictionary.
        model_name: The desired name for the Pydantic model.

    Returns:
        A Pydantic BaseModel class corresponding to the schema.
    """
    # Define JSON type to Python type mapping within the function scope
    TYPE_MAPPING: dict[str, type[Any]] = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        # 'object' and 'array' are handled recursively
        # 'null' is handled by Optional/Union logic
    }

    fields: dict[str, tuple[type[Any], Any]] = {}
    required_fields = set(schema.get("required", []))

    def get_type_annotation_and_field(
        prop_schema: dict[str, Any], prop_name: str
    ) -> tuple[type[Any], Any]:
        """Determines the type annotation and Pydantic Field configuration for a property."""
        field_args = {}
        description = prop_schema.get("description")
        if description:
            field_args["description"] = description

        default_value = prop_schema.get(
            "default", ...
        )  # Use Pydantic's Ellipsis for required

        prop_type: list[str] | str | None = prop_schema.get("type")
        final_annotation: Any

        if isinstance(prop_type, list):
            # Handle unions (like ["string", "null"])
            union_types = tuple(
                get_type_annotation_and_field(
                    {"type": t, "description": description}, prop_name
                )[0]
                for t in prop_type
                if t != "null"
            )
            if len(union_types) == 1:
                final_annotation = union_types[0]
            elif len(union_types) > 1:
                final_annotation = Union[union_types]  # type: ignore
            else:  # Only "null" was present
                final_annotation = Any  # Or handle as appropriate, maybe error?

            if "null" in prop_type:
                final_annotation = Optional[final_annotation]
                if (
                    default_value is ... and prop_name not in required_fields
                ):  # Set default to None only if not required
                    default_value = None

        elif prop_type == "object":
            nested_model_name = f"{model_name.rstrip('_')}_{prop_name.capitalize()}"
            # Recursively generate the nested model
            final_annotation = schema_to_model(
                prop_schema, nested_model_name
            )  # Changed from self._create_model

        elif prop_type == "array":
            items_schema = prop_schema.get("items")
            if not items_schema:
                item_type = Any  # Default to Any if items schema is missing
            else:
                # Recursively determine the type for array items
                item_type, _ = get_type_annotation_and_field(
                    items_schema, f"{prop_name}_item"
                )
            final_annotation = list[item_type]  # type: ignore

        else:
            # Map basic JSON types to Python types
            if isinstance(prop_type, str):
                final_annotation = TYPE_MAPPING.get(prop_type, Any)
            elif prop_type is None and "null" in prop_schema.get(
                "type", []
            ):  # Handle explicit null type
                final_annotation = Any  # Or Optional[Any] if appropriate
                if default_value is ... and prop_name not in required_fields:
                    default_value = None
            else:
                final_annotation = (
                    Any  # Default to Any if type is missing, not a string, or unhandled
                )

        is_required = prop_name in required_fields
        is_nullable = (
            (isinstance(prop_type, list) and "null" in prop_type)
            or prop_schema.get("type") == "null"  # Check schema type as well
        )

        # Adjust Optional and default value based on requirements and nullability
        if not is_required and default_value is ...:
            # If not required and no default provided, make it Optional unless already nullable
            if not is_nullable:
                final_annotation = Optional[final_annotation]
            default_value = None  # Set default to None for non-required fields without explicit default
        elif is_nullable and default_value is ... and prop_name in required_fields:
            # If required but nullable, Field(default=None) might be needed if Pydantic doesn't handle Optional[...] correctly for required
            pass  # Pydantic usually handles Optional[...] correctly, Ellipsis should suffice

        # Create Pydantic Field if needed (description or default provided)
        if field_args or default_value is not ...:
            # Pass Ellipsis explicitly if the field is required
            field_default = (
                default_value
                if default_value is not ...
                else (None if is_nullable else ...)
            )
            if (
                not is_required and field_default is ...
            ):  # Ensure non-required fields have a default
                field_default = None
            # If field_default is still Ellipsis here, it means it's required
            field_instance = Field(**field_args, default=field_default)  # type: ignore[call-arg]
        else:
            # If no description or default, use Ellipsis for required fields
            field_instance = ... if is_required else None

        return (final_annotation, field_instance)

    for prop_name, prop_schema in schema.get("properties", {}).items():
        fields[prop_name] = get_type_annotation_and_field(prop_schema, prop_name)

    # Create the main model using pydantic.create_model
    model: type[BaseModel] = create_model(model_name, **fields, __base__=BaseModel)  # type: ignore # Ensure it inherits from our BaseModel

    return cast(type[BaseModel], model)
