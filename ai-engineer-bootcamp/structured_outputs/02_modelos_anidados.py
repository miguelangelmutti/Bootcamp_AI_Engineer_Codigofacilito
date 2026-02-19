"""
Clase 4 - Snippet 02: Modelos Anidados
Composición de modelos Pydantic para estructuras complejas.
"""

from pydantic import BaseModel


class Ingrediente(BaseModel):
    nombre: str
    cantidad: float
    unidad: str


class Receta(BaseModel):
    titulo: str
    ingredientes: list[Ingrediente]
    pasos: list[str]


# --- Crear una receta ---
receta = Receta(
    titulo="Guacamole",
    ingredientes=[
        Ingrediente(nombre="aguacate", cantidad=3, unidad="piezas"),
        Ingrediente(nombre="cebolla", cantidad=0.5, unidad="piezas"),
        Ingrediente(nombre="cilantro", cantidad=1, unidad="manojo"),
        Ingrediente(nombre="limón", cantidad=2, unidad="piezas"),
        Ingrediente(nombre="sal", cantidad=1, unidad="pizca"),
    ],
    pasos=[
        "Cortar los aguacates por la mitad y sacar la pulpa.",
        "Picar la cebolla y el cilantro finamente.",
        "Mezclar todo en un bowl y machacar con un tenedor.",
        "Agregar jugo de limón y sal al gusto.",
    ],
)

print("Receta:")
print(receta)
print()

# Serialización a JSON
print("JSON:")
print(receta.model_dump_json(indent=2))
print()

# Acceder a datos anidados
print(f"Título: {receta.titulo}")
print(f"Número de ingredientes: {len(receta.ingredientes)}")
print(f"Primer ingrediente: {receta.ingredientes[0].nombre}")
print(f"Primer paso: {receta.pasos[0]}")
