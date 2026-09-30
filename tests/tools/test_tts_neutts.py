"""Regression tests for NeuTTS custom-backbone language propagation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestNeuTTSSynthHelper:
    def test_passes_language_to_neutts_constructor(self, tmp_path, monkeypatch):
        ref_audio = tmp_path / "ref.wav"
        ref_text = tmp_path / "ref.txt"
        output = tmp_path / "out.wav"
        ref_audio.write_bytes(b"wav")
        ref_text.write_text("reference text", encoding="utf-8")

        fake_tts = MagicMock()
        fake_tts.encode_reference.return_value = "encoded-ref"
        fake_tts.infer.return_value = [0.0, 0.0]
        fake_neutts_cls = MagicMock(return_value=fake_tts)
        monkeypatch.setitem(
            __import__("sys").modules,
            "neutts",
            SimpleNamespace(NeuTTS=fake_neutts_cls),
        )
        fake_soundfile = SimpleNamespace(
            write=lambda path, samples, samplerate: output.write_bytes(b"RIFF")
        )
        monkeypatch.setitem(__import__("sys").modules, "soundfile", fake_soundfile)
        monkeypatch.setattr(
            __import__("sys").modules["sys"],
            "argv",
            [
                "neutts_synth.py",
                "--text",
                "hello",
                "--out",
                str(output),
                "--ref-audio",
                str(ref_audio),
                "--ref-text",
                str(ref_text),
                "--model",
                "/models/custom.gguf",
                "--device",
                "cuda",
                "--language",
                "en-us",
            ],
        )

        from tools import neutts_synth

        neutts_synth.main()

        assert fake_neutts_cls.call_args.kwargs["language"] == "en-us"
        fake_tts.encode_reference.assert_called_once_with(str(ref_audio))
        fake_tts.infer.assert_called_once_with("hello", "encoded-ref", "reference text")
        assert output.exists()


class TestNeuTTSConfigPropagation:
    def test_passes_config_language_to_helper_command(self, tmp_path, monkeypatch):
        from tools import tts_tool_local

        captured = {}

        def fake_run(command, timeout):
            captured["command"] = command
            captured["timeout"] = timeout
            return SimpleNamespace(returncode=0, stdout="", stderr="OK: done")

        monkeypatch.setattr(tts_tool_local, "_run_helper", fake_run)
        monkeypatch.setattr(
            tts_tool_local,
            "_finalize_wav_output",
            lambda wav_path, output_path: output_path,
        )
        output = str(tmp_path / "out.wav")
        config = {
            "neutts": {
                "ref_audio": "/models/ref.wav",
                "ref_text": "/models/ref.txt",
                "model": "/models/custom.gguf",
                "device": "cuda",
                "language": "en-us",
                "timeout": 360,
            }
        }

        assert tts_tool_local._generate_neutts("hello", output, config) == output
        command = captured["command"]
        assert command[command.index("--language") + 1] == "en-us"
        assert command[command.index("--model") + 1] == "/models/custom.gguf"
        assert command[command.index("--device") + 1] == "cuda"
        assert captured["timeout"] == 360
