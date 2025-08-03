import os
import json
import mmap
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChunkManager:
    """
    Manages efficient retrieval of chunks from a large JSON file using
    an index and memory-mapping.
    """
    def __init__(self, json_path: str, index_path: str = "chunks.index.json"):
        self.json_path = json_path
        self.index_path = index_path
        self._file = None
        self._mmap = None

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Chunks JSON file not found at: {self.json_path}")

        if not os.path.exists(self.index_path) or os.path.getmtime(self.index_path) < os.path.getmtime(self.json_path):
            logger.info("Chunk index is outdated or does not exist. Building new index...")
            self._build_index()
        
        logger.info("Loading chunk index...")
        with open(self.index_path, 'r') as f:
            self.index = json.load(f)
        
        self._open_mmap()
        logger.info(f"ChunkManager initialized. {len(self.index)} chunks indexed.")

    def _open_mmap(self):
        """Opens the memory-mapped file."""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()
        self._file = open(self.json_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def _build_index(self):
        """
        Builds an index of chunk_id to (byte_offset, length) from the JSON file.
        This is a one-time operation or done when the source file changes.
        It robustly parses the JSON array without loading the whole file into memory.
        """
        index = {}
        with open(self.json_path, 'rb') as f:
            in_string = False
            brace_level = 0
            obj_start = -1

            # Find the start of the array
            f.seek(0)
            char = f.read(1)
            while char and char.isspace():
                char = f.read(1)
            if char != b'[':
                raise ValueError("A JSON array must start with '['")

            while True:
                pos = f.tell()
                char = f.read(1)
                if not char:
                    break

                if char == b'"':
                    # Basic string detection, ignores escaped quotes for simplicity
                    in_string = not in_string
                elif char == b'{' and not in_string:
                    if brace_level == 0:
                        obj_start = pos
                    brace_level += 1
                elif char == b'}' and not in_string:
                    brace_level -= 1
                    if brace_level == 0 and obj_start != -1:
                        obj_end = pos + 1
                        length = obj_end - obj_start
                        
                        f.seek(obj_start)
                        obj_bytes = f.read(length)
                        try:
                            chunk_data = json.loads(obj_bytes)
                            chunk_id = chunk_data.get("chunk_id")
                            if chunk_id:
                                index[str(chunk_id)] = (obj_start, length)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON object at offset {obj_start}")
                        
                        obj_start = -1
                        f.seek(obj_end)

        with open(self.index_path, 'w') as f:
            json.dump(index, f)
        logger.info(f"Successfully built and saved chunk index with {len(index)} entries.")

    def get_chunk(self, chunk_id: str) -> dict or None:
        """
        Retrieves a single chunk by its ID using the memory-mapped file.
        """
        chunk_id_str = str(chunk_id)
        if chunk_id_str not in self.index:
            return None
        
        offset, length = self.index[chunk_id_str]
        
        try:
            self._mmap.seek(offset)
            chunk_bytes = self._mmap.read(length)
            chunk_dict = json.loads(chunk_bytes.decode('utf-8'))
            return chunk_dict
        except (ValueError, AttributeError):
            self._open_mmap()
            self._mmap.seek(offset)
            chunk_bytes = self._mmap.read(length)
            chunk_dict = json.loads(chunk_bytes.decode('utf-8'))
            return chunk_dict
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id} via mmap: {e}")
            return None

    def __del__(self):
        """Ensure files are closed when the object is destroyed."""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()
