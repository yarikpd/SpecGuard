import pytest

from specguard.cli import service
from specguard.cli.config import APIConfig


def test_llm_check_uses_chat_api_and_closes(monkeypatch):
    calls = {}
    class Client:
        def __init__(self, **kwargs): calls.update(kwargs)
        def generate(self, messages, response_format):
            calls['format'] = response_format
            return '{"ok": true}'
        def close(self): calls['closed'] = True
    monkeypatch.setattr(service, 'OpenAILLMClient', Client)
    result = service.check_model_connection('llm', APIConfig('http://host/v1', 'chat', 'key'))
    assert calls['model'] == 'chat' and calls['api_key'] == 'key'
    assert calls['format'] == 'json' and calls['closed']
    assert 'JSON' in result


def test_embedding_check_validates_vector_and_closes_on_failure(monkeypatch):
    calls = {}
    class Client:
        def __init__(self, **kwargs): calls.update(kwargs)
        def embed_text(self, text): return [0.1, 0.2]
        def close(self): calls['closed'] = True
    monkeypatch.setattr(service, 'OpenAIEmbeddingClient', Client)
    assert 'размерности 2' in service.check_model_connection('embedding', APIConfig('http://host/v1', 'embed', 'secret'))
    assert calls['model_name'] == 'embed' and calls['api_key'] == 'secret' and calls['closed']
    Client.embed_text = lambda self, text: []
    calls['closed'] = False
    with pytest.raises(ValueError):
        service.check_model_connection('embedding', APIConfig('http://host/v1', 'embed'))
    assert calls['closed']


def test_keys_persist_independently_and_environment_overrides(tmp_path, monkeypatch):
    import stat
    from specguard.cli.config import load_config, save_api_key
    monkeypatch.setenv('SPECGUARD_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.delenv('SPECGUARD_LLM_API_KEY', raising=False)
    monkeypatch.delenv('SPECGUARD_EMBEDDING_API_KEY', raising=False)
    save_api_key('llm', 'llm-secret')
    save_api_key('embedding', 'embedding-secret')
    assert load_config().llm.api_key == 'llm-secret'
    assert load_config().embedding.api_key == 'embedding-secret'
    assert stat.S_IMODE((tmp_path / 'secrets.json').stat().st_mode) == 0o600
    monkeypatch.setenv('SPECGUARD_EMBEDDING_API_KEY', 'environment-secret')
    assert load_config().embedding.api_key == 'environment-secret'
    save_api_key('llm', '')
    assert load_config().llm.api_key == ''


def test_native_gemini_endpoint_explains_compatible_url():
    config = APIConfig('https://generativelanguage.googleapis.com/v1beta/interactions', 'gemini-model')
    with pytest.raises(ValueError, match='openai/chat/completions'):
        config.validate()
    assert config.base_url.endswith('/interactions')
    APIConfig('https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', 'gemini-model').validate()
