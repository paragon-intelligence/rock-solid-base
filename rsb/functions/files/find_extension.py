from rsb.functions.files.get_extension_agressivelly import (
    get_extension_agressivelly,
)
from rsb.functions.files.get_extension_from_filename import (
    get_extension_from_filename,
)
from rsb.functions.files.get_extension_from_url import (
    get_extension_from_url,
)
from rsb.functions.mime2ext import mime2ext


def find_extension(
    *,
    filename: str | None = None,
    content_type: str | None = None,
    contents: bytes | None = None,
    url: str | None = None,
) -> str:
    if filename and (ext := get_extension_from_filename(filename)):
        return ext
    if content_type and (ext := mime2ext(content_type)):
        return ext
    if contents and (ext := get_extension_agressivelly(contents)):
        return ext
    if url and (ext := get_extension_from_url(url)):
        return ext

    raise ValueError("Unable to determine the file extension.")
