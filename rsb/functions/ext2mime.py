import mimetypes


def ext2mime(extension: str) -> str:
    """
    Returns the MIME type associated with the given file extension.
    Returns None if the extension is unknown.
    """
    # Normalize the extension to include a leading dot
    if not extension.startswith("."):
        extension = "." + extension

    # Create a dummy filename with the given extension
    dummy_filename = f"dummy{extension}"
    mime_type, _ = mimetypes.guess_type(dummy_filename)
    if mime_type is None:
        raise ValueError("Unable to determine the MIME type.")

    return mime_type
