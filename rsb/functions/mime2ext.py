import mimetypes


def mime2ext(mime: str) -> str:
    """
    Returns the most common file extension for the given MIME type.
    Returns None if the MIME type is unknown.
    """
    _mime = mimetypes.guess_extension(mime, strict=False)
    if _mime is None:
        raise ValueError("Unable to determine the file extension.")

    return _mime.lstrip(".")
