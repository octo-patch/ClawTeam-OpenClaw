from __future__ import annotations

import pytest
from typer.testing import CliRunner

from clawteam.cli.commands import app
from clawteam.config import AgentPreset, AgentProfile, ClawTeamConfig, load_config, save_config
from clawteam.spawn.presets import generate_profile_from_preset, list_presets
from clawteam.spawn.profiles import apply_profile
from clawteam.team.manager import TeamManager


def test_generate_profile_from_builtin_preset():
    name, profile = generate_profile_from_preset("moonshot-cn", "claude")

    assert name == "claude-moonshot-cn"
    assert profile.agent == "claude"
    assert profile.model == "kimi-k2.5"
    assert profile.base_url == "https://api.moonshot.cn/anthropic"
    assert profile.api_key_env == "MOONSHOT_API_KEY"
    assert profile.env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "kimi-k2.5"


def test_generate_profile_from_openrouter_preset_for_multiple_clients():
    _, claude_profile = generate_profile_from_preset("openrouter", "claude")
    _, codex_profile = generate_profile_from_preset("openrouter", "codex")
    _, gemini_profile = generate_profile_from_preset("openrouter", "gemini")

    assert claude_profile.base_url == "https://openrouter.ai/api"
    assert claude_profile.api_key_env == "OPENROUTER_API_KEY"
    assert codex_profile.base_url == "https://openrouter.ai/api/v1"
    assert codex_profile.model == "openai/gpt-5.4"
    assert gemini_profile.base_url == "https://openrouter.ai/api"
    assert gemini_profile.model == "google/gemini-2.5-pro"


def test_generate_profile_from_minimax_global_preset():
    name, profile = generate_profile_from_preset("minimax-global", "claude")

    assert name == "claude-minimax-global"
    assert profile.agent == "claude"
    assert profile.model == "MiniMax-M3.1-Flash"
    assert profile.base_url == "https://api.minimax.io/anthropic"
    assert profile.api_key_env == "MINIMAX_API_KEY"
    assert profile.env["ANTHROPIC_MODEL"] == "MiniMax-M3.1-Flash"
    assert profile.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "MiniMax-M3.1-Flash"
    assert profile.env["API_TIMEOUT_MS"] == "3000000"
    assert profile.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_generate_profile_from_minimax_cn_preset():
    name, profile = generate_profile_from_preset("minimax-cn", "claude")

    assert name == "claude-minimax-cn"
    assert profile.agent == "claude"
    assert profile.model == "MiniMax-M3.1-Flash"
    assert profile.base_url == "https://api.minimaxi.com/anthropic"
    assert profile.api_key_env == "MINIMAX_API_KEY"
    assert profile.env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "MiniMax-M3.1-Flash"
    assert profile.env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "MiniMax-M3.1-Flash"


def test_generate_profile_from_minimax_m3_fallback_presets():
    _, global_profile = generate_profile_from_preset("minimax-global-m3", "claude")
    _, cn_profile = generate_profile_from_preset("minimax-cn-m3", "claude")

    assert global_profile.model == "MiniMax-M3"
    assert global_profile.base_url == "https://api.minimax.io/anthropic"
    assert cn_profile.model == "MiniMax-M3"
    assert cn_profile.base_url == "https://api.minimaxi.com/anthropic"
    assert global_profile.env["ANTHROPIC_MODEL"] == "MiniMax-M3"
    assert cn_profile.env["ANTHROPIC_MODEL"] == "MiniMax-M3"


@pytest.mark.parametrize("region,host", [("global", "minimax.io"), ("cn", "minimaxi.com")])
@pytest.mark.parametrize("suffix,model", [("", "MiniMax-M3.1-Flash"), ("-m3", "MiniMax-M3")])
def test_minimax_generated_profile_reaches_spawn_boundary(monkeypatch, region, host, suffix, model):
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-token")
    calls = []

    class RecordingBackend:
        def spawn(self, **kwargs):
            calls.append(kwargs)
            return "Agent spawned"

        def list_running(self):
            return []

    monkeypatch.setattr("clawteam.spawn.get_backend", lambda _: RecordingBackend())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["preset", "generate-profile", f"minimax-{region}{suffix}", "claude", "--name", "minimax"],
    )
    assert result.exit_code == 0, result.output
    cfg = load_config()
    cfg.default_profile = "minimax"
    save_config(cfg)
    TeamManager.create_team(name="demo", leader_name="leader", leader_id="leader001")
    result = runner.invoke(
        app,
        ["spawn", "subprocess", "--team", "demo", "--agent-name", "alice", "--no-workspace", "--task", "say hi"],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["command"] == ["claude", "--model", model]
    env = calls[0]["env"]
    assert env["ANTHROPIC_BASE_URL"] == f"https://api.{host}/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "fixture-token"
    for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL"):
        assert env[key] == model


def test_minimax_presets_in_builtin_list():
    presets = list_presets()
    assert "minimax-cn" in presets
    assert "minimax-global" in presets
    assert "minimax-cn-m3" in presets
    assert "minimax-global-m3" in presets
    _, source_cn = presets["minimax-cn"]
    _, source_global = presets["minimax-global"]
    assert source_cn == "builtin"
    assert source_global == "builtin"


def test_generate_profile_from_google_ai_studio_preset():
    name, profile = generate_profile_from_preset("google-ai-studio", "gemini")

    assert name == "gemini-google-ai-studio"
    assert profile.agent == "gemini"
    assert profile.model == "gemini-2.5-pro"
    assert profile.api_key_env == "GEMINI_API_KEY"


def test_local_preset_overrides_builtin(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAWTEAM_DATA_DIR", str(tmp_path / ".clawteam"))
    save_config(
        ClawTeamConfig(
            presets={
                "moonshot-cn": AgentPreset(
                    description="local override",
                    auth_env="LOCAL_MOONSHOT_KEY",
                    client_overrides={
                        "claude": AgentProfile(
                            agent="claude",
                            model="kimi-k3",
                            base_url="https://local.example/anthropic",
                        )
                    },
                )
            }
        )
    )

    preset, source = list_presets()["moonshot-cn"]
    assert source == "local"
    assert preset.auth_env == "LOCAL_MOONSHOT_KEY"

    _, profile = generate_profile_from_preset("moonshot-cn", "claude")
    assert profile.model == "kimi-k3"
    assert profile.base_url == "https://local.example/anthropic"
    assert profile.api_key_env == "LOCAL_MOONSHOT_KEY"


@pytest.mark.parametrize("model", ["MiniMax-M2.7", "MiniMax-M3", "custom-model"])
def test_local_minimax_preset_remains_authoritative(monkeypatch, tmp_path, model):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAWTEAM_DATA_DIR", str(tmp_path / ".clawteam"))
    save_config(
        ClawTeamConfig(
            presets={
                "minimax-global": AgentPreset(
                    description="saved legacy model",
                    auth_env="MINIMAX_API_KEY",
                    client_overrides={
                        "claude": AgentProfile(
                            agent="claude",
                            model=model,
                            base_url="https://saved.example/anthropic",
                        )
                    },
                )
            }
        )
    )

    preset, source = list_presets()["minimax-global"]
    _, profile = generate_profile_from_preset("minimax-global", "claude")

    assert source == "local"
    assert preset.description == "saved legacy model"
    assert profile.model == model
    assert profile.base_url == "https://saved.example/anthropic"
    command, env, _ = apply_profile(profile)
    assert command == ["claude", "--model", model]
    assert env["ANTHROPIC_BASE_URL"] == "https://saved.example/anthropic"


def test_preset_cli_copy_set_client_generate_and_bootstrap(tmp_path):
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CLAWTEAM_DATA_DIR": str(tmp_path / ".clawteam"),
    }

    result = runner.invoke(app, ["preset", "copy", "moonshot-cn", "moonshot-custom"], env=env)
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "preset",
            "set-client",
            "moonshot-custom",
            "claude",
            "--model",
            "kimi-k2.6",
            "--env",
            "ENABLE_TOOL_SEARCH=true",
        ],
        env=env,
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "preset",
            "generate-profile",
            "moonshot-custom",
            "claude",
            "--name",
            "claude-custom",
        ],
        env=env,
    )
    assert result.exit_code == 0

    result = runner.invoke(app, ["profile", "show", "claude-custom"], env=env)
    assert result.exit_code == 0
    assert "kimi-k2.6" in result.output
    assert "MOONSHOT_API_KEY" in result.output

    result = runner.invoke(
        app,
        ["preset", "bootstrap", "moonshot-custom", "--client", "claude", "--client", "kimi"],
        env=env,
    )
    assert result.exit_code == 0
    assert "claude-moonshot-custom" in result.output
    assert "kimi-moonshot-custom" in result.output

    cfg = load_config()
    assert "moonshot-custom" in cfg.presets
    assert "claude-custom" in cfg.profiles
    assert "claude-moonshot-custom" in cfg.profiles
    assert "kimi-moonshot-custom" in cfg.profiles
