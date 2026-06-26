# Semantic Model Router

Un enrutador semantico inteligente que analiza la complejidad de las consultas del usuario y las deriva de forma dinamica al modelo mas adecuado en coste y rendimiento. Esto permite optimizar costes operativos (reducciones de hasta un 70%) enviando prompts sencillos a modelos ultrarapidos y economicos, y reservando los modelos avanzados y costosos exclusivamente para razonamiento complejo y programacion.

## Arquitectura y Algoritmos de Enrutamiento

El sistema evalua el prompt de entrada mediante un pipeline logico de tres fases:

1.  **Analisis Lexico y Estructural (Regex):**
    *   Inspecciona la longitud del texto. Si es inferior a 25 caracteres, se asume complejidad baja y se enruta de inmediato a `gpt-4o-mini`.
    *   Busca expresiones regulares que indiquen desarrollo de software (`def`, `class`, `import`, `fn`, `struct`) o matematicas avanzadas. En caso afirmativo, se enruta de inmediato a `claude-3-5-sonnet`.
2.  **Similitud de N-Gramas de Caracteres (Similitud de Jaccard):**
    *   Si no hay coincidencia directa de regex, el prompt se compara contra intenciones de anclaje (coding_anchor, reasoning_anchor) usando n-gramas de nivel 3 (tri-grams).
    *   Este algoritmo calcula el solapamiento Jaccard, sirviendo de contingencia rapida, sin latencia y 100% offline frente a APIs de embeddings externas.
3.  **Integracion Modular de Embeddings (Opcional):**
    *   El enrutador esta preparado para importar dinamicamente el modulo `contrastive-embedding-trainer` del ecosistema `ai-core-infra`. Si esta disponible, permite realizar busquedas de similitud en un espacio vectorial local frente a plantillas de consulta clasificadas previamente.

## Modelos y Perfiles de Costo Soportados

El enrutador gestiona por defecto los siguientes perfiles de modelos:

*   **gpt-4o-mini:** Coste minimo, latencia muy baja, capacidad intelectual moderada (4/10). Asignado a tareas simples o chats rutinarios.
*   **gpt-4o:** Coste moderado, latencia media, capacidad analitica alta (8/10). Asignado a razonamientos generales de longitud moderada.
*   **claude-3-5-sonnet:** Coste alto, latencia alta, capacidad cognitiva superior (10/10). Reservado para generacion de codigo, optimizaciones complejas y auditorias.

## Requisitos e Instalacion

*   Python 3.8 o superior
*   NumPy
*   Pydantic

Para instalar las dependencias locales, ejecute:

```bash
pip install -r requirements.txt
```

## Ejecucion y Verificacion

1.  **Ejecutar Pruebas Automatizadas:**
    ```bash
    python -m unittest test_router.py
    ```
2.  **Ejecutar Script de Demostracion:**
    ```bash
    python example.py
    ```
    El script demostrara el enrutamiento de diferentes prompts e imprimira el desglose de costes y ahorros financieros proyectados frente a enviar todo al modelo mas caro.
