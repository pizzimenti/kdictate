"""Tests for PulseAudio/PipeWire input-device probing helpers."""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from kdictate.core.audio import (
    ACTIVATION_MIC_VOLUME_PERCENT,
    resolve_default_input_device,
    set_default_source_volume,
)


class AudioHelpersTest(unittest.TestCase):
    def test_resolve_default_input_device_uses_utf8_and_returns_description(self) -> None:
        side_effect = [
            subprocess.CompletedProcess(
                args=["pactl", "get-default-source"],
                returncode=0,
                stdout="alsa_input.pci-0000_00_1f.3.analog-stereo\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["pactl", "list", "sources"],
                returncode=0,
                stdout=(
                    "Name: alsa_input.pci-0000_00_1f.3.analog-stereo\n"
                    "Description: Built-in Audio Analog Stereo\n"
                ),
                stderr="",
            ),
        ]

        with mock.patch("kdictate.core.audio.subprocess.run", side_effect=side_effect) as run:
            self.assertEqual(
                resolve_default_input_device(),
                ("Built-in Audio Analog Stereo", True),
            )

        first_call = run.call_args_list[0]
        self.assertEqual(first_call.kwargs["encoding"], "utf-8")
        self.assertEqual(first_call.kwargs["errors"], "replace")

    def test_resolve_default_input_device_rejects_monitor_source(self) -> None:
        result = subprocess.CompletedProcess(
            args=["pactl", "get-default-source"],
            returncode=0,
            stdout="alsa_output.monitor\n",
            stderr="",
        )

        with mock.patch("kdictate.core.audio.subprocess.run", return_value=result):
            self.assertEqual(resolve_default_input_device(), ("alsa_output.monitor", False))

    def test_resolve_default_input_device_returns_unknown_when_pactl_fails(self) -> None:
        with mock.patch("kdictate.core.audio.subprocess.run", side_effect=OSError("missing pactl")):
            self.assertEqual(resolve_default_input_device(), ("unknown", False))

    def test_set_default_source_volume_invokes_pactl_with_default_percent(self) -> None:
        result = subprocess.CompletedProcess(
            args=["pactl", "set-source-volume", "@DEFAULT_SOURCE@", "91%"],
            returncode=0,
            stdout="",
            stderr="",
        )
        with mock.patch("kdictate.core.audio.subprocess.run", return_value=result) as run:
            self.assertTrue(set_default_source_volume())

        called_args = run.call_args_list[0].args[0]
        self.assertEqual(
            called_args,
            ["pactl", "set-source-volume", "@DEFAULT_SOURCE@", f"{ACTIVATION_MIC_VOLUME_PERCENT}%"],
        )

    def test_set_default_source_volume_returns_false_on_pactl_failure(self) -> None:
        result = subprocess.CompletedProcess(
            args=["pactl", "set-source-volume", "@DEFAULT_SOURCE@", "91%"],
            returncode=1,
            stdout="",
            stderr="no such source\n",
        )
        with mock.patch("kdictate.core.audio.subprocess.run", return_value=result):
            self.assertFalse(set_default_source_volume())

    def test_set_default_source_volume_returns_false_when_pactl_missing(self) -> None:
        with mock.patch("kdictate.core.audio.subprocess.run", side_effect=OSError("missing pactl")):
            self.assertFalse(set_default_source_volume())
