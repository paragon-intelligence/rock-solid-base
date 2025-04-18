from rsb.functions.files.get_extension_from_filename import (
    get_extension_from_filename,
)


def get_extension_from_url(url: str) -> str:
    """Extracts the file extension from a URL."""
    from urllib.parse import urlparse

    parsed_url = urlparse(url)
    path = parsed_url.path
    if not path:
        raise ValueError("The URL does not contain a path.")
    name = path.split("/")[-1]
    if "." not in name:
        raise ValueError("The URL does not contain a file extension.")

    return get_extension_from_filename(name)
