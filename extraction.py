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
    output_format = os.getenv("MARKER_OUTPUT_FORMAT", "chunks").strip() or "chunks"

    # Flags: LLM + inline math ON by default, OCR OFF by default
    use_force_ocr = _env_bool("MARKER_FORCE_OCR", False)
    use_llm = _env_bool("MARKER_USE_LLM", True)
    redo_inline_math = _env_bool("MARKER_REDO_INLINE_MATH", True)
    drop_repeated = _env_bool("MARKER_DROP_REPEATED_TEXT", True)

    # Extra args passthrough (advanced users)
    extra_args = shlex.split(os.getenv("MARKER_EXTRA_ARGS", ""))

    # Choose CLI based on input type and optional chunk-convert mode
    use_chunk_convert = _env_bool("MARKER_USE_CHUNK_CONVERT", False)
    is_file = os.path.isfile(source_path)
    if use_chunk_convert and not is_file:
        cli = "marker_chunk_convert"
    else:
        cli = "marker_single" if is_file else "marker"

    # Build command
    command = [
        cli,
        source_path,
        "--output_dir", output_dir,
        "--output_format", output_format,
    ]

    # Only include LLM if requested AND key present
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if use_llm and not gemini_key:
        print("[extraction] MARKER_USE_LLM=true but GEMINI_API_KEY is missing; disabling LLM for this run.")
        use_llm = False
    if gemini_key:
        command.extend(["--gemini_api_key", gemini_key])

    if use_force_ocr:
        command.append("--force_ocr")
    if use_llm:
        command.append("--use_llm")
    if redo_inline_math:
        command.append("--redo_inline_math")
    if drop_repeated:
        command.append("--drop_repeated_text")

    # Optional: page range (single-file mode)
    page_range = os.getenv("MARKER_PAGE_RANGE", "").strip()
    if page_range and is_file:
        command.extend(["--page_range", page_range])

    # Optional: workers (folder mode)
    workers = os.getenv("MARKER_WORKERS", "").strip()
    if workers and not is_file and not use_chunk_convert:
        command.extend(["--workers", workers])

    # Optional: disable image extraction
    if _env_bool("MARKER_DISABLE_IMAGE_EXTRACTION", False):
        command.append("--disable_image_extraction")

    # Optional: strip existing OCR text
    if _env_bool("MARKER_STRIP_EXISTING_OCR", False):
        command.append("--strip_existing_ocr")

    # Optional: converter class override and force layout block
    converter_cls = os.getenv("MARKER_CONVERTER_CLS", "").strip()
    if converter_cls:
        command.extend(["--converter_cls", converter_cls])
    force_layout_block = os.getenv("MARKER_FORCE_LAYOUT_BLOCK", "").strip()
    if force_layout_block:
        command.extend(["--force_layout_block", force_layout_block])

    # Optional: specify LLM service backend
    llm_service = os.getenv("MARKER_LLM_SERVICE", "").strip()
    if llm_service:
        command.extend(["--llm_service", llm_service])

    if extra_args:
        command.extend(extra_args)

    # Select Gemini model (safe: pass via env). Default to user's requested model.
    gemini_model = os.getenv("GEMINI_MODEL", os.getenv("MARKER_GEMINI_MODEL", "gemini-1.5-flash-8b")).strip()
    child_env = os.environ.copy()
    if gemini_model:
        child_env["GEMINI_MODEL"] = gemini_model
    # Optional: multi-GPU chunk convert environment variables
    if use_chunk_convert:
        num_devices = os.getenv("NUM_DEVICES")
        num_workers = os.getenv("NUM_WORKERS")
        if num_devices:
            child_env["NUM_DEVICES"] = num_devices
        if num_workers:
            child_env["NUM_WORKERS"] = num_workers

    print("Running Marker with:")
    print(" ", " ".join(shlex.quote(str(p)) for p in command))
    if use_llm:
        print(f" Using LLM service: {llm_service or 'gemini'} | model: {gemini_model}")

    t0 = time.time()
    result = subprocess.run(command, capture_output=True, text=True, env=child_env)
    elapsed = time.time() - t0

    print("\n==== STDOUT ====\n")
    print(result.stdout)
    print("\n==== STDERR ====\n")
    print(result.stderr)
    print(f"\nElapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    run_marker()
