import os
import subprocess
from dotenv import load_dotenv

def run_marker():
    # Load your API key from .env file
    load_dotenv()
    
    # Define your directories (adjust as per your project!)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "Source_Documents")
    output_dir = os.path.join(base_dir, "New folder")
    
    # Build the Marker command
    command = [
        "marker",
        source_dir,
        "--output_dir", output_dir,
        "--force_ocr",
        "--use_llm",
        "--redo_inline_math",
        "--drop_repeated_text",
        "--output_format", "chunks",
        "--gemini_api_key", os.getenv("GEMINI_API_KEY"),  # Make sure key is loaded     
    ]

    # Run the Marker command and display output
    result = subprocess.run(command, capture_output=True, text=True)

    print("\n==== STDOUT ====\n")
    print(result.stdout)
    print("\n==== STDERR ====\n")
    print(result.stderr)

if __name__ == "__main__":
    run_marker()
