"""
Clase 4 - Práctica Paso 1: Definir el Contrato
Modelos Pydantic con Field descriptions para recetas de cocina.
"""

import json

from pydantic import BaseModel, Field


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


# --- Inspeccionar el JSON Schema ---
schema = Receta.model_json_schema()
print("JSON Schema de Receta:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
