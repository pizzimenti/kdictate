"""Tests for PulseAudio/PipeWire input-device probing helpers."""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from kdictate.core.audio import (
    MIN_MIC_VOLUME_PERCENT,
    ensure_default_source_volume,
    read_default_source_volume,
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
            ["pactl", "set-source-volume", "@DEFAULT_SOURCE@", f"{MIN_MIC_VOLUME_PERCENT}%"],
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


# Verbatim `pactl get-source-volume @DEFAULT_SOURCE@` output.
_PACTL_VOLUME_OUTPUT = (
    "Volume: front-left: 59637 /  91% / -2.46 dB,   "
    "front-right: 59637 /  91% / -2.46 dB\n"
    "        balance 0.00\n"
)


def _volume_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["pactl", "get-source-volume", "@DEFAULT_SOURCE@"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


class ReadSourceVolumeTest(unittest.TestCase):
    def test_parses_the_percentage_from_real_pactl_output(self) -> None:
        with mock.patch(
            "kdictate.core.audio.subprocess.run",
            return_value=_volume_result(_PACTL_VOLUME_OUTPUT),
        ):
            self.assertEqual(read_default_source_volume(), 91)

    def test_returns_none_when_no_percentage_is_present(self) -> None:
        with mock.patch(
            "kdictate.core.audio.subprocess.run",
            return_value=_volume_result("Volume: unknown\n"),
        ):
            self.assertIsNone(read_default_source_volume())

    def test_returns_none_when_pactl_fails(self) -> None:
        with mock.patch(
            "kdictate.core.audio.subprocess.run",
            return_value=_volume_result("", returncode=1),
        ):
            self.assertIsNone(read_default_source_volume())


class EnsureSourceVolumeTest(unittest.TestCase):
    """The daemon may rescue a drifted-down gain; it may not impose one."""

    def test_a_healthy_level_is_left_completely_alone(self) -> None:
        # The regression this whole change exists to prevent. Pinning a fixed
        # level on every activation drove already-healthy sources into
        # clipping, which costs the VAD the dynamic range it needs.
        with mock.patch(
            "kdictate.core.audio.read_default_source_volume", return_value=91
        ):
            with mock.patch(
                "kdictate.core.audio.set_default_source_volume"
            ) as setter:
                self.assertTrue(ensure_default_source_volume(50))
        setter.assert_not_called()

    def test_a_level_exactly_at_the_floor_is_left_alone(self) -> None:
        with mock.patch(
            "kdictate.core.audio.read_default_source_volume", return_value=50
        ):
            with mock.patch(
                "kdictate.core.audio.set_default_source_volume"
            ) as setter:
                self.assertTrue(ensure_default_source_volume(50))
        setter.assert_not_called()

    def test_a_drifted_level_is_raised_to_the_floor(self) -> None:
        # The failure PR #10 was written for: the source had drifted to 40%
        # and no speech crossed the VAD threshold.
        with mock.patch(
            "kdictate.core.audio.read_default_source_volume", return_value=40
        ):
            with mock.patch(
                "kdictate.core.audio.set_default_source_volume", return_value=True
            ) as setter:
                self.assertTrue(ensure_default_source_volume(50))
        setter.assert_called_once_with(50)

    def test_zero_floor_never_touches_the_microphone(self) -> None:
        with mock.patch("kdictate.core.audio.read_default_source_volume") as reader:
            with mock.patch(
                "kdictate.core.audio.set_default_source_volume"
            ) as setter:
                self.assertTrue(ensure_default_source_volume(0))
        reader.assert_not_called()
        setter.assert_not_called()

    def test_unknown_level_is_not_guessed_at(self) -> None:
        with mock.patch(
            "kdictate.core.audio.read_default_source_volume", return_value=None
        ):
            with mock.patch(
                "kdictate.core.audio.set_default_source_volume"
            ) as setter:
                self.assertFalse(ensure_default_source_volume(50))
        setter.assert_not_called()
