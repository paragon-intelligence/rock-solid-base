import mimetypes

import magic


def get_extension_agressivelly(contents: bytes) -> str:
    def _detect_mime_type_manually(content: bytes) -> str:
        return "application/octet-stream"

    # Detect MIME type
    try:
        mime_type = magic.Magic(mime=True).from_buffer(contents)
    except Exception:
        mime_type = _detect_mime_type_manually(contents)

    # Convert MIME type to extension
    extension = mimetypes.guess_extension(mime_type)
    if extension is None:
        extension = ".bin"  # Default to .bin for unknown MIME types

    return extension
