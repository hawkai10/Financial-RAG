import os
import sys
import time
import shlex
import subprocess
from dotenv import load_dotenv

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def run_marker():
    # Load your API key from .env file
    load_dotenv()

    # Resolve input/output paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(base_dir, "Source_Documents")
    default_output = os.path.join(base_dir, "New folder")

    source_path = os.getenv("MARKER_INPUT_PATH", default_source)
    output_dir = os.getenv("MARKER_OUTPUT_DIR", default_output)

    # Flags (tune for speed)
    use_force_ocr = _env_bool("MARKER_FORCE_OCR", True)
    use_llm = _env_bool("MARKER_USE_LLM", True)
    redo_inline_math = _env_bool("MARKER_REDO_INLINE_MATH", True)
    drop_repeated = _env_bool("MARKER_DROP_REPEATED_TEXT", True)

    # Extra args passthrough (advanced users)
    extra_args = shlex.split(os.getenv("MARKER_EXTRA_ARGS", ""))

    # Build command
    command = [
        "marker",
        source_path,
        "--output_dir", output_dir,
        "--output_format", "chunks",
        "--gemini_api_key", os.getenv("GEMINI_API_KEY", ""),
    ]

    if use_force_ocr:
        command.append("--force_ocr")
    if use_llm:
        command.append("--use_llm")
    if redo_inline_math:
        command.append("--redo_inline_math")
    if drop_repeated:
        command.append("--drop_repeated_text")

    if extra_args:
        command.extend(extra_args)

    print("Running Marker with:")
    print(" ", " ".join(shlex.quote(str(p)) for p in command))

    t0 = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - t0

    print("\n==== STDOUT ====\n")
    print(result.stdout)
    print("\n==== STDERR ====\n")
    print(result.stderr)
    print(f"\nElapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    run_marker()
