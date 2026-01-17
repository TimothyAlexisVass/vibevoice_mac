import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "models", "VibeVoice-7B")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
INFER = os.path.join(ROOT, "demo", "inference_from_file.py")
OUTDIR = os.path.join(ROOT, "outputs")
TMPDIR = "/tmp/vibevoice_chunks"
VOICE = "MariaMolina"

MAX_CHARS = 400
PREFIX = "Speaker 1: "
MAX_LENGTH_TIMES = 6

TEXT = """
Guarda las porciones directamente en el congelador para mantenerlas frescas por más tiempo.
Puedes cocinar varias preparaciones al mismo tiempo y ahorrar energía.
Saca los ingredientes de la bolsa justo antes de usarlos.
Es importante cerrar bien y apilar correctamente en el congelador para optimizar el espacio.
Sofríe cebolla y ajo al inicio para dar más sabor al plato.
Añade caldo vegetal para una base más aromática y ligera.
Esta receta es ideal para personas con alergias e intolerancias o para dietas especiales.
Vierte todo a una olla o crockpot y deja que se cocine lentamente.
Esta combinación es una de mis favoritas para el día a día.
Prepara una variedad de snacks saludables en un plis plas.
Ajusta las especias según tu gusto personal.
Conseguimos esa cremosidad que dan los anacardos sin usar lácteos.
Antes de cerrar la bolsa, vamos a eliminar todo el aire posible.
Cuando terminamos, repetimos bolsa etiquetada y la colocamos con las demás.
Cierra bien para que aguante nuestra bolsa durante la congelación.
Si no es con kale lo puedes reemplazar por espinacas o acelgas sin problema.
Añade sal y hierbas aromáticas al final para potenciar el sabor.
La sopa de kale y alubias aplanamos y reservamos antes de congelar.
Agregamos las especias y hierbas aromáticas, en este caso tomillo y laurel, pero puedes reemplazarlos por otra aromática a gusto.
Es importante tener siempre listo un fondo casero en el congelador.
Puedes usar alubias negras o frijoles negros o rojos también.
Las lentejas rojas no están cocidas para que tengan mejor textura en el momento de la cocción.
Rectificamos el sabor añadiendo las especias poco a poco.
Esta base se añade a esta receta a la hora de la elaboración.
Así conseguimos que nos quede lista en 30 minutos en una olla o crockpot.
Dejamos todas las bolsas listas antes de empezar la semana.
""".strip()

os.makedirs(TMPDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

limit = MAX_CHARS

sentences = [s.strip() for s in TEXT.splitlines() if s.strip()]
for s in sentences:
    if len(s) > limit:
        raise RuntimeError(f"A single sentence exceeds {limit} chars (cannot chunk without splitting): {s[:120]}...")

chunks = []
current = ""

for s in sentences:
    if not current:
        current = s
        continue
    candidate = current + " " + s
    if len(candidate) <= limit:
        current = candidate
    else:
        chunks.append(current)
        current = s

if current:
    chunks.append(current)

items = []
for i, chunk in enumerate(chunks, start=1):
    path = f"{TMPDIR}/chunk_{i}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREFIX + chunk)
    items.append((path, chunk))

stage = 1
while True:
    print(f"\n===== Stage {stage}")
    for txt, chunk in items:
        print(f"input file: {os.path.basename(txt)}")
        print(f'text to generate: "{PREFIX + chunk}"')
        subprocess.run(
            [
                VENV_PY,
                INFER,
                "--model_path",
                MODEL,
                "--txt_path",
                txt,
                "--speaker_names",
                VOICE,
                "--output_dir",
                OUTDIR,
                "--device",
                "mps",
                "--max_length_times",
                str(MAX_LENGTH_TIMES),
            ],
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            check=True,
        )
        txt_basename = os.path.splitext(os.path.basename(txt))[0]
        base_out = os.path.join(OUTDIR, f"{txt_basename}_generated.wav")
        stage_out = os.path.join(OUTDIR, f"{stage}_{txt_basename}_generated.wav")
        if os.path.exists(base_out):
            os.replace(base_out, stage_out)
    stage += 1
