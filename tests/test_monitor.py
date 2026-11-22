"""Мониторинг нагрузки не должен запускать процессы (иначе на Windows мигают окна)."""
from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from viralcut import system_stats as ss


class WindowsNoSpawnTest(unittest.TestCase):
    def setUp(self):
        ss._GPU_CACHE.update({"value": None, "ts": 0.0})
        ss._GPU_CACHE.pop("legacy", None)
        ss._MEM_CACHE.update({"value": None, "ts": 0.0})
        ss._LAST_CPU.clear()
        ss._LAST_VALUE[0] = 0.0
        ss._LAST_TS[0] = 0.0
        ss._GPU_FAIL_TS[0] = 0.0

    def test_snapshot_does_not_spawn_processes(self):
        spawned = []

        def boom(*args, **kwargs):
            spawned.append(args)
            raise AssertionError("не должен запускаться процесс")

        with mock.patch.object(ss, "_platform", return_value="win32"), \
             mock.patch.object(ss, "_cpu_windows", return_value=(10, 20)), \
             mock.patch.object(ss, "_gpu_windows_pdh", return_value={"percent": 7, "name": "GPU"}), \
             mock.patch.object(ss.subprocess, "run", side_effect=boom), \
             mock.patch.object(ss.subprocess, "Popen", side_effect=boom):
            data = ss.snapshot()

        self.assertEqual(spawned, [], "на Windows мониторинг не должен ничего запускать")
        self.assertEqual(data["gpu"]["percent"], 7)
        self.assertIsInstance(data["ram"], dict)

    def test_gpu_fallback_is_rate_limited(self):
        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1, "", "нет nvidia-smi")

        with mock.patch.object(ss, "_platform", return_value="win32"), \
             mock.patch.object(ss, "_cpu_windows", return_value=(10, 20)), \
             mock.patch.object(ss, "_gpu_windows_pdh", return_value={"percent": None, "name": ""}), \
             mock.patch.object(ss.subprocess, "run", side_effect=fake_run):
            for _ in range(6):
                ss.gpu(force=True)

        self.assertLessEqual(len(calls), 1,
                             "запасной nvidia-smi должен вызываться не чаще раза в 30 с")

    def test_macos_path_used_for_darwin(self):
        ss._GPU_CACHE.update({"value": None, "ts": 0.0})
        with mock.patch.object(ss, "_platform", return_value="darwin"), \
             mock.patch.object(ss, "_gpu_macos",
                               return_value={"percent": 42, "name": "Apple M1"}) as gpu_mock:
            data = ss.gpu(force=True)
        gpu_mock.assert_called_once()
        self.assertEqual(data["percent"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
