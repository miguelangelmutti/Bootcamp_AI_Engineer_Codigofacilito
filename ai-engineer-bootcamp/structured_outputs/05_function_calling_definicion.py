"""
Clase 4 - Snippet 05: Function Calling - Definición
Cómo definir tools (funciones) para el modelo.
"""

import json

# --- Definición de una tool ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "buscar_producto",
            "description": "Busca productos en el catálogo por nombre o categoría.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Término de búsqueda del producto.",
                    },
                    "categoria": {
                        "type": "string",
                        "enum": ["electrónica", "ropa", "hogar", "deportes"],
                        "description": "Categoría del producto.",
                    },
                    "precio_maximo": {
                        "type": "number",
                        "description": "Precio máximo en pesos mexicanos.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

# Inspeccionar la estructura
print("Definición de tools:")
print(json.dumps(tools, indent=2, ensure_ascii=False))
print()

# Acceder a partes específicas
funcion = tools[0]["function"]
print(f"Nombre: {funcion['name']}")
print(f"Descripción: {funcion['description']}")
print(f"Parámetros requeridos: {funcion['parameters']['required']}")
print(f"Todos los parámetros: {list(funcion['parameters']['properties'].keys())}")
