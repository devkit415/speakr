"""Upload size limits with chunking on (#533).

Chunking lifts the per-type max_file_size_mb cap, as documented, but never the
MAX_CONTENT_LENGTH ceiling. Werkzeug refuses a larger request, with or without
Content-Length, and the upload handler checks the ceiling again whatever the
chunking setting.
"""
import io
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.app import app, db
from src.models import Recording, SystemSetting
from tests.test_cov_recordings_write import _mk_user, _login, _upload_mocks, _cleanup

BOUNDARY = "speakrtestboundary"


def _chunking_on():
    config = MagicMock()
    config.enabled = True
    config.source = "env"
    config.mode = "size"
    config.limit_value = 20
    return patch("src.audio_chunking.get_effective_chunking_config", return_value=config)


def _multipart(payload, filename="big.mp3"):
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/mpeg\r\n\r\n"
    ).encode() + payload + f"\r\n--{BOUNDARY}--\r\n".encode()


def _post(client, payload, *, content_length=True):
    body = _multipart(payload)
    kwargs = {"content_type": f"multipart/form-data; boundary={BOUNDARY}"}
    if content_length:
        return client.post("/upload", data=body, **kwargs)
    # Chunked transfer encoding: no Content-Length, the server reads to the end.
    return client.post("/upload", input_stream=io.BytesIO(body),
                       environ_overrides={"wsgi.input_terminated": True}, **kwargs)


@pytest.fixture
def user_client():
    with app.app_context():
        user = _mk_user("up_chunk")
        client = app.test_client()
        _login(client, user)
        yield user, client
        Recording.query.filter_by(user_id=user.id).delete()
        db.session.commit()
        _cleanup(user)


def _limits(policy_mb):
    return patch.object(SystemSetting, "get_setting",
                        side_effect=lambda k, d=None: policy_mb if k == "max_file_size_mb" else d)


@pytest.mark.parametrize("content_length", [True, False])
def test_the_request_ceiling_holds_with_chunking_on(user_client, content_length):
    user, client = user_client
    staging = os.path.join(app.config["UPLOAD_FOLDER"], f"stg_{uuid.uuid4().hex[:6]}")
    original = app.config["MAX_CONTENT_LENGTH"]
    try:
        app.config["MAX_CONTENT_LENGTH"] = 4096
        with _upload_mocks(staging), _limits(500), _chunking_on(), \
             patch("src.api.recordings.chunking_service", MagicMock()):
            resp = _post(client, b"\x00" * 16384, content_length=content_length)
    finally:
        app.config["MAX_CONTENT_LENGTH"] = original
    assert resp.status_code == 413, resp.get_data(as_text=True)
    assert Recording.query.filter_by(user_id=user.id).count() == 0


def test_chunking_still_lifts_the_per_type_cap(user_client):
    user, client = user_client
    staging = os.path.join(app.config["UPLOAD_FOLDER"], f"stg_{uuid.uuid4().hex[:6]}")
    with _upload_mocks(staging), _limits(1), _chunking_on(), \
         patch("src.api.recordings.chunking_service", MagicMock()):
        resp = _post(client, b"\x00" * (2 * 1024 * 1024))
    assert resp.status_code in (200, 202), resp.get_data(as_text=True)
    assert Recording.query.filter_by(user_id=user.id).count() == 1


def test_the_per_type_cap_applies_with_chunking_off(user_client):
    user, client = user_client
    staging = os.path.join(app.config["UPLOAD_FOLDER"], f"stg_{uuid.uuid4().hex[:6]}")
    with _upload_mocks(staging), _limits(1), \
         patch("src.api.recordings.chunking_service", None):
        resp = _post(client, b"\x00" * (2 * 1024 * 1024))
    assert resp.status_code == 413
    assert Recording.query.filter_by(user_id=user.id).count() == 0
