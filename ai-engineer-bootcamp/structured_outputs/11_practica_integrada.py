"""
Clase 4 - Práctica 11: Integración Pydantic + Function Calling + Structured Outputs

Asistente de soporte para una tienda en línea que combina los 3 conceptos:
  1. Pydantic → define contratos de datos y argumentos de tools
  2. Function Calling → el modelo decide qué función llamar (consultar pedido,
     buscar producto, calcular envío)
  3. Structured Outputs → la respuesta final se parsea a un objeto Pydantic
     con formato predecible para producción
"""

import json
import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "openai/gpt-oss-120b"


# =============================================
# PASO 1: Contratos Pydantic
# =============================================

# --- Modelos de dominio ---

class EstatusPedido(str, Enum):
    preparando = "preparando"
    enviado = "enviado"
    entregado = "entregado"


class Producto(BaseModel):
    id: str = Field(description="Identificador único del producto.")
    nombre: str = Field(description="Nombre del producto.")
    precio: float = Field(description="Precio en pesos mexicanos.")
    stock: int = Field(description="Unidades disponibles.")


class Pedido(BaseModel):
    id: str = Field(description="Número de pedido.")
    cliente: str = Field(description="Nombre del cliente.")
    productos: list[str] = Field(description="Lista de nombres de productos.")
    total: float = Field(description="Total del pedido en pesos mexicanos.")
    estatus: EstatusPedido = Field(description="Estatus actual del pedido.")


class CostoEnvio(BaseModel):
    destino: str = Field(description="Ciudad de destino.")
    costo: float = Field(description="Costo de envío en pesos mexicanos.")
    dias_estimados: int = Field(description="Días estimados de entrega.")


# --- Modelo para Structured Outputs (respuesta final) ---

class RespuestaSoporte(BaseModel):
    resumen: str = Field(description="Resumen breve de lo que se encontró o resolvió.")
    datos: str = Field(description="Datos relevantes encontrados, formateados como texto.")
    siguiente_paso: str = Field(description="Acción sugerida al cliente.")


# =============================================
# PASO 2: Datos simulados y funciones locales
# =============================================

CATALOGO: list[Producto] = [
    Producto(id="LAP-001", nombre="Laptop Pro 15", precio=24999, stock=5),
    Producto(id="LAP-002", nombre="Laptop Air 13", precio=18999, stock=12),
    Producto(id="AUD-001", nombre="Audífonos Bluetooth", precio=1299, stock=30),
    Producto(id="TAB-001", nombre="Tablet 10 pulgadas", precio=8499, stock=8),
    Producto(id="MON-001", nombre="Monitor 27 4K", precio=7999, stock=3),
]

PEDIDOS: list[Pedido] = [
    Pedido(
        id="PED-2024-001",
        cliente="María González",
        productos=["Laptop Pro 15", "Audífonos Bluetooth"],
        total=26298,
        estatus=EstatusPedido.enviado,
    ),
    Pedido(
        id="PED-2024-002",
        cliente="Carlos López",
        productos=["Tablet 10 pulgadas"],
        total=8499,
        estatus=EstatusPedido.preparando,
    ),
]

TARIFAS_ENVIO = {
    "cdmx": CostoEnvio(destino="CDMX", costo=99, dias_estimados=2),
    "guadalajara": CostoEnvio(destino="Guadalajara", costo=149, dias_estimados=3),
    "monterrey": CostoEnvio(destino="Monterrey", costo=149, dias_estimados=3),
    "cancún": CostoEnvio(destino="Cancún", costo=199, dias_estimados=5),
}


def consultar_pedido(numero_pedido: str) -> str:
    """Busca un pedido por su número."""
    for pedido in PEDIDOS:
        if pedido.id.lower() == numero_pedido.lower():
            return pedido.model_dump_json()
    return json.dumps({"error": f"No se encontró el pedido {numero_pedido}"})


def buscar_producto(termino: str) -> str:
    """Busca productos en el catálogo por nombre."""
    termino_lower = termino.lower()
    resultados = [p for p in CATALOGO if termino_lower in p.nombre.lower()]
    if not resultados:
        return json.dumps({"error": f"No se encontraron productos con '{termino}'"})
    return json.dumps([p.model_dump() for p in resultados], ensure_ascii=False)


def calcular_envio(ciudad: str) -> str:
    """Calcula el costo de envío a una ciudad."""
    tarifa = TARIFAS_ENVIO.get(ciudad.lower())
    if tarifa:
        return tarifa.model_dump_json()
    ciudades_disponibles = [t.destino for t in TARIFAS_ENVIO.values()]
    return json.dumps(
        {"error": f"Ciudad '{ciudad}' no disponible. Ciudades: {ciudades_disponibles}"},
        ensure_ascii=False,
    )


FUNCIONES = {
    "consultar_pedido": consultar_pedido,
    "buscar_producto": buscar_producto,
    "calcular_envio": calcular_envio,
}


# =============================================
# PASO 3: Definición de tools para el modelo
# =============================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "consultar_pedido",
            "description": "Consulta el estatus y detalles de un pedido por su número.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numero_pedido": {
                        "type": "string",
                        "description": "Número de pedido, ej: PED-2024-001",
                    },
                },
                "required": ["numero_pedido"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_producto",
            "description": "Busca productos en el catálogo por nombre o categoría.",
            "parameters": {
                "type": "object",
                "properties": {
                    "termino": {
                        "type": "string",
                        "description": "Término de búsqueda del producto.",
                    },
                },
                "required": ["termino"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_envio",
            "description": "Calcula el costo y tiempo de envío a una ciudad de México.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ciudad": {
                        "type": "string",
                        "description": "Ciudad de destino: CDMX, Guadalajara, Monterrey, Cancún.",
                    },
                },
                "required": ["ciudad"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "Eres un asistente de soporte de TecnoShop, una tienda de tecnología en línea. "
    "Ayuda a los clientes consultando pedidos, buscando productos y calculando envíos. "
    "Usa las herramientas disponibles para obtener la información que necesites."
)


# =============================================
# PASO 4: Flujo completo — Function Calling
# =============================================

def ejecutar_tool_call(tool_call) -> str:
    """Ejecuta una tool call y retorna el resultado."""
    nombre = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"  [Tool: {nombre}({json.dumps(args, ensure_ascii=False)})]")

    funcion = FUNCIONES.get(nombre)
    if not funcion:
        return json.dumps({"error": f"Función desconocida: {nombre}"})
    return funcion(**args)


def chat_soporte(mensaje_usuario: str) -> str:
    """Ejecuta el flujo completo de function calling."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": mensaje_usuario},
    ]

    # Primera llamada: el modelo decide si usar tools
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        tools=tools,
    )

    mensaje_asistente = response.choices[0].message

    # Si no necesita tools, responde directamente
    if not mensaje_asistente.tool_calls:
        return mensaje_asistente.content

    # Ejecutar la tool call
    tool_call = mensaje_asistente.tool_calls[0]
    resultado = ejecutar_tool_call(tool_call)

    # Enviar resultado al modelo para respuesta final
    messages.append(mensaje_asistente)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": resultado,
    })

    response_final = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        tools=tools,
    )

    return response_final.choices[0].message.content


# =============================================
# PASO 5: Structured Outputs — Respuesta final
# =============================================

def chat_soporte_estructurado(mensaje_usuario: str) -> RespuestaSoporte:
    """
    Mismo flujo de function calling, pero la respuesta final
    se parsea con Structured Outputs a RespuestaSoporte.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": mensaje_usuario},
    ]

    # Primera llamada: function calling
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        tools=tools,
    )

    mensaje_asistente = response.choices[0].message

    # Ejecutar tool call si existe
    if mensaje_asistente.tool_calls:
        tool_call = mensaje_asistente.tool_calls[0]
        resultado = ejecutar_tool_call(tool_call)

        messages.append(mensaje_asistente)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": resultado,
        })

    # Segunda llamada: Structured Outputs para respuesta formateada
    messages.append({
        "role": "user",
        "content": "Ahora responde al cliente con la información obtenida.",
    })

    response_final = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.2,
        response_format=RespuestaSoporte,
        messages=messages,
    )

    return response_final.choices[0].message.parsed


# =============================================
# EJECUCIÓN
# =============================================

def imprimir_respuesta(resp: RespuestaSoporte) -> None:
    print(f"\n  Resumen: {resp.resumen}")
    print(f"  Datos: {resp.datos}")
    print(f"  Siguiente paso: {resp.siguiente_paso}")
    print(f"\n  JSON:\n{resp.model_dump_json(indent=2)}")


# --- Prueba 1: Consultar pedido ---
print("=" * 60)
print("  PRUEBA 1: Consultar pedido")
print("=" * 60)
resp = chat_soporte_estructurado("Hola, quiero saber el estatus de mi pedido PED-2024-001")
imprimir_respuesta(resp)

# --- Prueba 2: Buscar producto ---
print("\n" + "=" * 60)
print("  PRUEBA 2: Buscar producto")
print("=" * 60)
resp = chat_soporte_estructurado("¿Tienen laptops disponibles? ¿Cuánto cuestan?")
imprimir_respuesta(resp)

# --- Prueba 3: Calcular envío ---
print("\n" + "=" * 60)
print("  PRUEBA 3: Calcular envío")
print("=" * 60)
resp = chat_soporte_estructurado("¿Cuánto cuesta el envío a Guadalajara y cuánto tarda?")
imprimir_respuesta(resp)

# --- Comparación: sin vs con Structured Outputs ---
print("\n" + "=" * 60)
print("  COMPARACIÓN: Respuesta libre vs estructurada")
print("=" * 60)
pregunta = "Quiero rastrear mi pedido PED-2024-002"
print(f"\n  Pregunta: {pregunta}")

print("\n  --- Respuesta libre (Function Calling solo) ---")
resp_libre = chat_soporte(pregunta)
print(f"  {resp_libre[:200]}...")

print("\n  --- Respuesta estructurada (Function Calling + Structured Outputs) ---")
resp_struct = chat_soporte_estructurado(pregunta)
imprimir_respuesta(resp_struct)
