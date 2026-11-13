"""Bound request-body size before multipart parsing can spool an upload."""

import json
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_upload_bytes: int):
        self.app = app
        self.max_body_bytes = max_upload_bytes + 64 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/analyze"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._reject(send, "invalid_content_length")
                return
            if declared_length > self.max_body_bytes:
                await self._reject(send, "upload_size_limit_exceeded")
                return

        buffered = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            buffered.extend(message.get("body", b""))
            if len(buffered) > self.max_body_bytes:
                await self._reject(send, "upload_size_limit_exceeded")
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": bytes(buffered),
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(send: Send, reason: str) -> None:
        body = json.dumps(
            {
                "request_id": str(uuid4()),
                "model_version": "unavailable",
                "status": "unsupported_or_unsuitable",
                "input_suitability": "rejected",
                "abstention_reason": reason,
                "message": "Unsupported image: Unable to verify that this is a breast ultrasound image.",
                "classification": None,
                "probabilities": None,
                "explanation": None,
                "segmentation": None,
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
