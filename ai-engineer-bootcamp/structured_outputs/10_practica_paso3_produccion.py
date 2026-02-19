"""
Clase 4 - Práctica Paso 3: Output para Producción
Extracción de receta con formato listo para uso en producción.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "openai/gpt-oss-120b"


# --- Modelos ---
class Ingrediente(BaseModel):
    nombre: str = Field(description="Nombre del ingrediente.")
    cantidad: float = Field(description="Cantidad numérica del ingrediente.")
    unidad: str = Field(description="Unidad de medida: piezas, gramos, ml, cucharadas, etc.")


class Receta(BaseModel):
    titulo: str = Field(description="Nombre del platillo.")
    tiempo_minutos: int = Field(description="Tiempo total de preparación en minutos.")
    dificultad: str = Field(description="Nivel de dificultad: fácil, media o difícil.")
    ingredientes: list[Ingrediente] = Field(description="Lista de ingredientes.")
    pasos: list[str] = Field(description="Pasos de preparación en orden.")


# --- Texto desordenado ---
texto_receta = """
mira para los chilaquiles verdes necesitas como medio kilo de totopos
o si quieres los haces con tortillas pero es más trabajo, también unos
3 tomates verdes, un diente de ajo, cebolla al gusto, crema y queso
fresco para servir. Ah y chile serrano como 2 o 3 dependiendo qué tan
picoso te guste. Primero hierves los tomates con el chile y el ajo como
15 minutos, luego los licuas con un poco de agua. En una sartén con
aceite fríes la salsa como 5 minutos, le pones sal, y ya al final
echas los totopos y mezclas bien. Se sirven con crema, queso y cebolla.
En total como en 30 minutos los tienes listos, es bastante fácil.
"""

# --- Extracción ---
response = client.beta.chat.completions.parse(
    model=MODEL,
    temperature=0.2,
    response_format=Receta,
    messages=[
        {
            "role": "system",
            "content": (
                "Eres un chef experto. Extrae la receta del texto del usuario "
                "en el formato estructurado solicitado. Convierte cantidades "
                "informales a números precisos. IMPORTANTE: Usa exactamente "
                "los nombres de campo del schema (nombre, cantidad, unidad)."
            ),
        },
        {"role": "user", "content": texto_receta},
    ],
)

receta = response.choices[0].message.parsed

# --- Formato para producción ---
print("=" * 50)
print(f"  {receta.titulo.upper()}")
print("=" * 50)
print(f"  Tiempo: {receta.tiempo_minutos} min | Dificultad: {receta.dificultad}")
print("-" * 50)

print("\n  INGREDIENTES:")
for ing in receta.ingredientes:
    print(f"    - {ing.cantidad} {ing.unidad} de {ing.nombre}")

print(f"\n  PREPARACION:")
for i, paso in enumerate(receta.pasos, 1):
    print(f"    {i}. {paso}")

print("\n" + "=" * 50)

# --- JSON completo para API/base de datos ---
print("\nJSON para almacenamiento:")
print(receta.model_dump_json(indent=2))
