from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import cpu
from zarr.core.keys import is_valid_node_key

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response

    from zarr.abc.store import ByteRequest, Store
    from zarr.core.array import Array
    from zarr.core.group import Group

__all__ = ["CorsOptions", "HTTPMethod", "serve_node", "serve_store"]


class CorsOptions(TypedDict):
    allow_origins: list[str]
    allow_methods: list[str]


HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]


def _parse_range_header(range_header: str) -> ByteRequest | None:
    """Parse an HTTP Range header into a ByteRequest.

    Parameters
    ----------
    range_header : str
        The value of the Range header, e.g. ``"bytes=0-99"`` or ``"bytes=-100"``.

    Returns
    -------
    ByteRequest or None
        A ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``,
        or ``None`` if the header cannot be parsed.
    """
    if not range_header.startswith("bytes="):
        return None
    range_spec = range_header[len("bytes=") :]
    try:
        if range_spec.startswith("-"):
            # suffix request: bytes=-N
            suffix = int(range_spec[1:])
            return SuffixByteRequest(suffix=suffix)
        parts = range_spec.split("-", 1)
        if len(parts) != 2:
            return None
        start_str, end_str = parts
        start = int(start_str)
        if end_str == "":
            # offset request: bytes=N-
            return OffsetByteRequest(offset=start)
        # range request: bytes=N-M (HTTP end is inclusive, ByteRequest end is exclusive)
        end = int(end_str) + 1
        return RangeByteRequest(start=start, end=end)
    except ValueError:
        return None


async def _get_response(store: Store, path: str, byte_range: ByteRequest | None = None) -> Response:
    """Fetch a key from the store and return an HTTP response."""
    from starlette.responses import Response

    proto = cpu.buffer_prototype
    content_type = "application/json" if path.endswith("zarr.json") else "application/octet-stream"

    buf = await store.get(path, proto, byte_range=byte_range)
    if buf is None:
        return Response(status_code=404)

    status_code = 206 if byte_range is not None else 200
    return Response(content=buf.to_bytes(), status_code=status_code, media_type=content_type)


async def _handle_request(request: Request) -> Response:
    """Handle a request, optionally filtering by node validity."""
    from starlette.responses import Response

    store: Store = request.app.state.store
    node: Array[Any] | Group | None = request.app.state.node
    prefix: str = request.app.state.prefix
    path = request.path_params.get("path", "")

    # If serving a node, validate the key before touching the store.
    if node is not None and not is_valid_node_key(node, path):
        return Response(status_code=404)

    # Resolve the full store key by prepending the node's prefix.
    store_key = f"{prefix}/{path}" if prefix else path

    if request.method == "PUT":
        body = await request.body()
        buf = cpu.buffer_prototype.buffer.from_bytes(body)
        await store.set(store_key, buf)
        return Response(status_code=204)

    range_header = request.headers.get("range")
    byte_range: ByteRequest | None = None
    if range_header is not None:
        byte_range = _parse_range_header(range_header)
        if byte_range is None:
            return Response(status_code=416)

    return await _get_response(store, store_key, byte_range)


def _make_starlette_app(
    *,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
) -> Starlette:
    """Create a Starlette app with the request handler."""
    try:
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.routing import Route
    except ImportError as e:
        raise ImportError(
            "The zarr server requires the 'starlette' package. "
            "Install it with: pip install zarr[server]"
        ) from e

    if methods is None:
        methods = {"GET"}

    app = Starlette(
        routes=[Route("/{path:path}", _handle_request, methods=list(methods))],
    )

    if cors_options is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_options["allow_origins"],
            allow_methods=cors_options["allow_methods"],
        )
    return app


def serve_store(
    store: Store,
    *,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
) -> Starlette:
    """Create a Starlette ASGI app that serves every key in a zarr ``Store``.

    Parameters
    ----------
    store : Store
        The zarr store to serve.
    methods : set of HTTPMethod, optional
        The HTTP methods to accept. Defaults to ``{"GET"}``.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.

    Returns
    -------
    Starlette
        An ASGI application.
    """
    app = _make_starlette_app(methods=methods, cors_options=cors_options)
    app.state.store = store
    app.state.node = None
    app.state.prefix = ""
    return app


def serve_node(
    node: Array[Any] | Group,
    *,
    methods: set[HTTPMethod] | None = None,
    cors_options: CorsOptions | None = None,
) -> Starlette:
    """Create a Starlette ASGI app that serves only the keys belonging to a
    zarr ``Array`` or ``Group``.

    For an ``Array``, the served keys are the metadata document(s) and all
    chunk (or shard) keys whose coordinates fall within the array's grid.

    For a ``Group``, the served keys are the group's own metadata plus any
    path that resolves through the group's members to a valid array metadata
    document or chunk key.

    Requests for keys outside this set receive a 404 response, even if the
    underlying store contains data at that path.

    Parameters
    ----------
    node : Array or Group
        The zarr array or group to serve.
    methods : set of HTTPMethod, optional
        The HTTP methods to accept. Defaults to ``{"GET"}``.
    cors_options : CorsOptions, optional
        If provided, CORS middleware will be added with the given options.

    Returns
    -------
    Starlette
        An ASGI application.
    """
    app = _make_starlette_app(methods=methods, cors_options=cors_options)
    app.state.store = node.store_path.store
    app.state.node = node
    app.state.prefix = node.store_path.path
    return app
