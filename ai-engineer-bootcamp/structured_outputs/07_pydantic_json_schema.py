"""
Clase 4 - Snippet 07: Pydantic → JSON Schema
Generar JSON Schemas automáticamente desde modelos Pydantic.
"""

import json

from pydantic import BaseModel, Field


class BuscarProductoArgs(BaseModel):
    query: str = Field(description="Término de búsqueda del producto.")
    categoria: str | None = Field(
        default=None,
        description="Categoría del producto: electrónica, ropa, hogar, deportes.",
    )
    precio_maximo: float | None = Field(
        default=None,
        description="Precio máximo en pesos mexicanos.",
    )


# --- Generar JSON Schema ---
schema = BuscarProductoArgs.model_json_schema()

print("JSON Schema generado automáticamente:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
print()

# --- Construir definición de tool a partir del schema ---
tool_definition = {
    "type": "function",
    "function": {
        "name": "buscar_producto",
        "description": "Busca productos en el catálogo.",
        "parameters": schema,
    },
}

print("Definición de tool completa:")
print(json.dumps(tool_definition, indent=2, ensure_ascii=False))
