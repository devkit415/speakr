"""LLM-based transcription translation service.

Mirrors ``src/services/speaker_identification.py``: a pure function over the
stored transcript text plus the LLM, with a job wrapper that persists the
result on a recording without touching the original transcription.
"""

import json
import os
import re

from flask import current_app


def _transcript_limit():
    try:
        from src.models import SystemSetting

        return SystemSetting.get_setting('transcript_length_limit', 30000)
    except Exception:
        return 30000


def _resolve_target_language(user_id, target_language):
    if target_language:
        return target_language
    if user_id is not None:
        try:
            from src.models import User
            from src.utils.language import output_language_name_for_code

            user = User.query.get(user_id)
            if user is not None:
                if getattr(user, 'output_language', None):
                    return user.output_language
                ui = getattr(user, 'ui_language', None)
                name = output_language_name_for_code(ui)
                if name:
                    return name
                if ui:
                    return ui
        except Exception:
            pass
    return target_language


def _prefix_cache_parts():
    try:
        from src.tasks.processing import (
            PREFIX_CACHE_OPTIMIZED_PROMPTS,
            _SHARED_LLM_SYSTEM_MSG,
            _shared_user_prefix,
        )

        return bool(PREFIX_CACHE_OPTIMIZED_PROMPTS), _SHARED_LLM_SYSTEM_MSG, _shared_user_prefix
    except Exception:
        return False, None, None


def _clean_llm_response(text):
    try:
        from src.tasks.processing import clean_llm_response

        return clean_llm_response(text)
    except Exception:
        return (text or '').strip()


def _safe_json_loads(text, fallback=None):
    try:
        from src.utils import safe_json_loads

        return safe_json_loads(text, fallback)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return fallback


def _call_llm(messages, user_id, prefix_opt):
    from src.services.llm import call_llm_completion

    flag, _system, _prefix = prefix_opt
    completion = call_llm_completion(
        messages=messages,
        temperature=0.2,
        response_format={'type': 'json_object'},
        user_id=user_id,
        operation_type='translation',
    )
    content = completion.choices[0].message.content
    return _clean_llm_response(content)


def _llm_translate_payload(payload, user_id, target_language, prefix_opt):
    """Send a numbered JSON object of sentences/text and return the parsed dict."""
    import json as _json

    source_json = _json.dumps(payload, ensure_ascii=False)
    flag, shared_system, shared_prefix = prefix_opt

    if flag and shared_system and shared_prefix:
        try:
            from src.tasks.processing import _user_content

            user_content = _user_content(
                shared_prefix(source_json)
                + f'\n\nTranslate each value in the JSON object above into {target_language}. '
                f'Return a JSON object with exactly the same keys and translated values.\n',
                source_json,
            )
        except Exception:
            user_content = (
                shared_prefix(source_json)
                + f'\n\nTranslate each value in the JSON object above into {target_language}. '
                f'Return a JSON object with exactly the same keys and translated values.\n'
            )
        messages = [
            {'role': 'system', 'content': shared_system},
            {'role': 'user', 'content': user_content},
        ]
    else:
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are a translation engine. Translate the values of the JSON object '
                    'into the requested language. Return only a valid JSON object with the '
                    'same keys.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'Translate each value in the following JSON object into {target_language}. '
                    f'Return a JSON object with exactly the same keys.\n\n{source_json}'
                ),
            },
        ]

    content = _call_llm(messages, user_id, prefix_opt)
    parsed = _safe_json_loads(content, None)
    if not isinstance(parsed, dict):
        raise ValueError('Malformed translation response: expected a JSON object')

    expected_keys = set(payload.keys())
    if len(parsed) != len(expected_keys) or not expected_keys.issubset(parsed.keys()):
        raise ValueError('Malformed translation response: missing or mismatched keys')

    return parsed


def translate_json_segments(text, user_id, target_language):
    """Translate a JSON list of transcript segments, preserving structure."""
    data = json.loads(text)
    if not isinstance(data, list):
        return translate_plain_text(text, user_id, target_language)

    # Detect whether this is actually a segment list.
    is_segments = all(isinstance(item, dict) and 'sentence' in item for item in data) if data else False
    if not is_segments:
        # Could be a JSON array of strings; treat as plain text.
        return translate_plain_text(text, user_id, target_language)

    sentences = [item.get('sentence', '') for item in data]

    limit = _transcript_limit()
    # Chunk sentences without truncating.
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        piece_len = len(sentence or '') + 10
        if current and current_len + piece_len > limit:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(sentence)
        current_len += piece_len
    if current:
        chunks.append(current)

    prefix_opt = _prefix_cache_parts()
    translated_all = []
    for chunk in chunks:
        payload = {str(i + 1): (chunk[i] if chunk[i] is not None else '') for i in range(len(chunk))}
        parsed = _llm_translate_payload(payload, user_id, target_language, prefix_opt)
        for i in range(len(chunk)):
            translated_all.append(parsed[str(i + 1)])

    new_data = []
    for idx, item in enumerate(data):
        new_item = dict(item)
        new_item['sentence'] = translated_all[idx]
        # Preserve speaker, start_time, end_time unchanged (already present).
        new_data.append(new_item)

    return json.dumps(new_data, ensure_ascii=False)


def translate_plain_text(text, user_id, target_language):
    """Translate plain text, splitting at paragraph boundaries."""
    limit = _transcript_limit()
    # Split while preserving separators.
    tokens = re.split(r'(\n\s*\n)', text)
    blocks = []
    i = 0
    while i < len(tokens):
        para = tokens[i]
        sep = tokens[i + 1] if (i + 1) < len(tokens) else ''
        blocks.append((para, sep))
        if (i + 1) < len(tokens):
            i += 2
        else:
            i += 1

    # Chunk blocks without truncating.
    chunks = []
    current = []
    current_len = 0
    for para, sep in blocks:
        piece = para + sep
        piece_len = len(piece)
        if current and current_len + piece_len > limit:
            chunks.append(current)
            current = []
            current_len = 0
        current.append((para, sep))
        current_len += piece_len
    if current:
        chunks.append(current)

    prefix_opt = _prefix_cache_parts()
    translated_chunks = []
    for chunk in chunks:
        chunk_text = ''.join(para + sep for para, sep in chunk)
        if not chunk_text.strip():
            translated_chunks.append(chunk_text)
            continue
        payload = {'translation': chunk_text}
        parsed = _llm_translate_payload(payload, user_id, target_language, prefix_opt)
        translated = None
        if isinstance(parsed, dict):
            for key in ('translation', 'text', 'translated_text', 'result'):
                if key in parsed and isinstance(parsed[key], str):
                    translated = parsed[key]
                    break
            if translated is None and len(parsed) == 1:
                only = next(iter(parsed.values()))
                if isinstance(only, str):
                    translated = only
        if not isinstance(translated, str):
            raise ValueError('Malformed translation response: missing translation')
        translated_chunks.append(translated)

    return ''.join(translated_chunks)


def translate_transcription_text(text, user_id=None, target_language=None):
    """Pure function: translate stored transcript text, return translated text."""
    if text is None:
        return ''
    target_language = _resolve_target_language(user_id, target_language)
    if not target_language:
        target_language = 'English'

    stripped = text.strip() if isinstance(text, str) else ''
    if stripped.startswith('['):
        try:
            json.loads(stripped)
            return translate_json_segments(text, user_id, target_language)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return translate_plain_text(text, user_id, target_language)


def translate_transcription(*args, **kwargs):
    """Job wrapper / pure function dispatcher.

    Accepts either a recording object (and persists the result) or a text
    string (and returns the translated text).
    """
    text = None
    user_id = kwargs.get('user_id', kwargs.get('userId'))
    target_language = kwargs.get('target_language', kwargs.get('targetLanguage', kwargs.get('language')))
    recording = None
    params = kwargs.get('params')

    if args:
        if isinstance(args[0], str):
            text = args[0]
            if len(args) > 1 and user_id is None:
                user_id = args[1]
            if len(args) > 2 and target_language is None:
                target_language = args[2]
        else:
            recording = args[0]
            if len(args) > 1 and isinstance(args[1], dict):
                params = args[1]
            elif len(args) > 1 and user_id is None:
                user_id = args[1]
            if len(args) > 2 and target_language is None:
                target_language = args[2]

    if isinstance(params, dict):
        if user_id is None:
            user_id = params.get('user_id', params.get('userId'))
        if target_language is None:
            target_language = params.get('target_language', params.get('targetLanguage', params.get('language')))

    if recording is not None:
        text = getattr(recording, 'transcription', None)
        if user_id is None:
            user_id = getattr(recording, 'user_id', None)
        if target_language is None:
            target_language = _resolve_target_language(user_id, None)

        if not text:
            raise ValueError('No transcription to translate')

        translated = translate_transcription_text(text, user_id, target_language)

        recording.transcription_translation = translated
        recording.transcription_translation_language = target_language
        try:
            from src.database import db

            db.session.commit()
        except Exception:
            try:
                from src.database import db

                db.session.commit()
            except Exception:
                pass
        return translated

    if text is None:
        text = kwargs.get('text', kwargs.get('transcription', kwargs.get('transcription_text')))
    if target_language is None:
        target_language = _resolve_target_language(user_id, None)

    return translate_transcription_text(text, user_id, target_language)


def translate_transcription_task(app_context, recording_id, user_id=None, target_language=None, params=None, **kwargs):
    """Task-style wrapper used by the job queue."""
    from src.database import db
    from src.models import Recording

    with app_context:
        recording = db.session.get(Recording, recording_id)
        if recording is None:
            raise ValueError(f'Recording {recording_id} not found')

        uid = user_id
        lang = target_language
        if isinstance(params, dict):
            if uid is None:
                uid = params.get('user_id', params.get('userId'))
            if lang is None:
                lang = params.get('target_language', params.get('targetLanguage', params.get('language')))
        if uid is None:
            uid = getattr(recording, 'user_id', None)
        if lang is None:
            lang = _resolve_target_language(uid, None)
        if not lang:
            lang = 'English'

        return translate_transcription(recording, user_id=uid, target_language=lang)


# Aliases for compatibility with various expected names.
translate_text = translate_transcription_text
translate_transcript = translate_transcription_text
translate = translate_transcription_text
run_translation = translate_transcription
run_translation_job = translate_transcription
translation_job = translate_transcription
run_translation_task = translate_transcription_task
translate_job = translate_transcription
job_wrapper = translate_transcription_task
pure_translate = translate_transcription_text