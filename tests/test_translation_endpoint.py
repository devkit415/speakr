import os, sys, uuid, json

import pytest

from src.app import app, db
from src.models import User, Recording, ProcessingJob
from src.api import auth as auth_mod

GOOD_PW = "Sup3rSecret!"


@pytest.fixture(autouse=True)
def _disable_csrf():
    prev = app.config.get('WTF_CSRF_ENABLED')
    app.config['WTF_CSRF_ENABLED'] = False
    yield
    app.config['WTF_CSRF_ENABLED'] = prev


@pytest.fixture
def app_ctx():
    with app.app_context():
        yield


@pytest.fixture
def cleanup_users(app_ctx):
    created = {"usernames": set(), "emails": set()}
    yield created
    q = User.query.filter(
        db.or_(
            User.username.in_(created["usernames"] or [""]),
            User.email.in_(created["emails"] or [""]),
        )
    )
    for u in q.all():
        db.session.delete(u)
    db.session.commit()


def _mk_user(cleanup, **extra):
    username = "u_" + uuid.uuid4().hex[:14]
    email = uuid.uuid4().hex[:10] + "@example.com"
    hashed = auth_mod.bcrypt.generate_password_hash(GOOD_PW).decode('utf-8')
    u = User(username=username, email=email, password=hashed, **extra)
    db.session.add(u)
    db.session.commit()
    cleanup["usernames"].add(username)
    cleanup["emails"].add(email)
    return u


def _mk_recording(user, transcription="Hello world this is a test transcript", **extra):
    r = Recording(user_id=user.id, transcription=transcription, status='COMPLETED', **extra)
    db.session.add(r)
    db.session.commit()
    return r


def _login(client, user):
    with client.session_transaction() as s:
        s['_user_id'] = str(user.id)
        s['_fresh'] = True


def test_view_access_denied(cleanup_users):
    owner = _mk_user(cleanup_users)
    other = _mk_user(cleanup_users)
    rec = _mk_recording(owner)
    client = app.test_client()
    _login(client, other)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={})
    assert resp.status_code == 404


def test_edit_denied_for_viewer(cleanup_users, monkeypatch):
    import src.app as appmod
    monkeypatch.setattr(appmod, 'ENABLE_INTERNAL_SHARING', True)
    owner = _mk_user(cleanup_users)
    viewer = _mk_user(cleanup_users)
    rec = _mk_recording(owner)
    from src.models import InternalShare
    share = InternalShare(recording_id=rec.id, owner_id=owner.id, shared_with_user_id=viewer.id, can_edit=False)
    db.session.add(share)
    db.session.commit()
    client = app.test_client()
    _login(client, viewer)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={})
    assert resp.status_code == 403


def test_missing_transcript(cleanup_users):
    owner = _mk_user(cleanup_users)
    rec = _mk_recording(owner, transcription="short")
    client = app.test_client()
    _login(client, owner)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'translation_no_transcription' in str(data)


def test_error_transcript(cleanup_users):
    owner = _mk_user(cleanup_users)
    rec = _mk_recording(owner, transcription="Transcription failed: boom")
    client = app.test_client()
    _login(client, owner)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={})
    assert resp.status_code == 400


def test_unconfigured_model(cleanup_users, monkeypatch):
    import src.services.llm as llmmod
    monkeypatch.setattr(llmmod, 'TRANSLATION_MODEL_CONFIGURED', False)
    owner = _mk_user(cleanup_users)
    rec = _mk_recording(owner)
    client = app.test_client()
    _login(client, owner)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={})
    assert resp.status_code == 503


def test_invalid_language(cleanup_users):
    owner = _mk_user(cleanup_users)
    rec = _mk_recording(owner)
    client = app.test_client()
    _login(client, owner)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={'target_language': 'Klingon'})
    assert resp.status_code == 400
    assert resp.get_json().get('code') == 'translation_invalid_language' or 'translation_invalid_language' in str(resp.get_json())


def test_missing_target_resolves_output_language(cleanup_users):
    owner = _mk_user(cleanup_users, output_language='Spanish')
    rec = _mk_recording(owner)
    client = app.test_client()
    _login(client, owner)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={})
    assert resp.status_code == 200
    assert 'job_id' in resp.get_json()


def test_missing_target_resolves_interface_language(cleanup_users):
    owner = _mk_user(cleanup_users, ui_language='es', output_language=None)
    rec = _mk_recording(owner)
    client = app.test_client()
    _login(client, owner)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'job_id' in data
    job = db.session.get(ProcessingJob, data['job_id'])
    import json as _j
    params = _j.loads(job.params)
    assert params.get('target_language') == 'Spanish'


def test_success_includes_job_id(cleanup_users):
    owner = _mk_user(cleanup_users)
    rec = _mk_recording(owner)
    client = app.test_client()
    _login(client, owner)
    resp = client.post(f'/recording/{rec.id}/translate_transcription', json={'target_language': 'Spanish'})
    assert resp.status_code == 200
    assert 'job_id' in resp.get_json()


def test_dedup(cleanup_users):
    owner = _mk_user(cleanup_users)
    rec = _mk_recording(owner)
    client = app.test_client()
    _login(client, owner)
    r1 = client.post(f'/recording/{rec.id}/translate_transcription', json={'target_language': 'Spanish'})
    r2 = client.post(f'/recording/{rec.id}/translate_transcription', json={'target_language': 'Spanish'})
    assert r1.get_json()['job_id'] == r2.get_json()['job_id']


def test_download_translation_uses_the_same_template_as_the_transcript(cleanup_users):
    """?translation=1 formats the translated segments, not the raw JSON blob."""
    owner = _mk_user(cleanup_users)
    rec = _mk_recording(
        owner,
        transcription=json.dumps([
            {"speaker": "Alice", "sentence": "Hello there", "start_time": 0.0, "end_time": 1.5},
            {"speaker": "Bob", "sentence": "Hi back", "start_time": 1.5, "end_time": 3.0},
        ]),
        transcription_translation=json.dumps([
            {"speaker": "Alice", "sentence": "Hola", "start_time": 0.0, "end_time": 1.5},
            {"speaker": "Bob", "sentence": "Adios", "start_time": 1.5, "end_time": 3.0},
        ]),
        transcription_translation_language='Spanish',
    )
    client = app.test_client()
    _login(client, owner)

    resp = client.get(f'/recording/{rec.id}/download/transcript?format=txt&translation=1')

    assert resp.status_code == 200
    body = resp.data.decode()
    assert '[Alice]: Hola' in body
    assert '[Bob]: Adios' in body
    assert '"sentence"' not in body
