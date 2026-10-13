"""模型管理的单元测试。

下载逻辑用本地 HTTP 服务来测，不碰外网 —— 否则"网络好不好"会变成
测试结果的一部分。真正的外网来源正确性由 SHA-256 校验兜住。
"""

from __future__ import annotations

import hashlib
import http.server
import threading
import zipfile
from pathlib import Path

import pytest

from imgcrop import models
from imgcrop.models import ModelEntry, ModelError


# ----------------------------------------------------------------------
# 本地 HTTP 服务：支持 Range，便于测断点续传与镜像回退
# ----------------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的接口
        server: _TestServer = self.server  # type: ignore[assignment]
        server.hits.append((self.path, self.headers.get("Range")))

        if self.path in server.fail_paths:
            self.send_error(500, "boom")
            return
        payload = server.routes.get(self.path)
        if payload is None:
            self.send_error(404, "not found")
            return

        start = 0
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start = int(range_header[len("bytes="):].split("-")[0])
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            )
        else:
            self.send_response(200)
        body = payload[start:]
        # 声明完整长度，但首次请求可以只发一半 —— 用来模拟"连接被掐断"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if self.path in server.truncate_once:
            server.truncate_once.discard(self.path)
            self.wfile.write(body[: max(1, len(body) // 2)])
            return
        self.wfile.write(body)

    def log_message(self, *args):  # 静音
        return


class _TestServer(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.routes: dict[str, bytes] = {}
        self.fail_paths: set[str] = set()
        self.truncate_once: set[str] = set()
        self.hits: list[tuple[str, str | None]] = []
        self.base_url = f"http://127.0.0.1:{self.server_address[1]}"
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()

    def path_for(self, entry: ModelEntry) -> str:
        return f"/{entry.repo}/resolve/main/{entry.filename}"


@pytest.fixture
def server():
    instance = _TestServer()
    try:
        yield instance
    finally:
        instance.stop()


def fake_entry(payload: bytes, filename: str = "fake.onnx") -> ModelEntry:
    return ModelEntry(
        key="fake",
        label="测试模型",
        filename=filename,
        repo="test/repo",
        sha256=hashlib.sha256(payload).hexdigest(),
        purpose="测试用",
    )


# 2.6 MB：必须大于下载分块（1 MiB），否则"取消/续传"这类测试根本来不及介入 ——
# 55 KB 的文件一次 read 就读完了，取消信号永远赶不上。
PAYLOAD = b"onnx-bytes-" * 240_000


class TestRegistry:
    def test_both_models_registered(self):
        assert set(models.MODELS) == {"isnet-general-use", "efficientsam-ti"}

    def test_hashes_are_real_sha256(self):
        for entry in models.MODELS.values():
            assert len(entry.sha256) == 64
            int(entry.sha256, 16)  # 必须是合法十六进制

    def test_every_entry_has_download_urls(self):
        for entry in models.MODELS.values():
            urls = entry.urls()
            assert len(urls) == len(models.DEFAULT_MIRRORS)
            assert all(entry.filename in url for url in urls)

    def test_license_safe_sources(self):
        """两个来源都必须是 Apache-2.0 的可分发权重（避开 AGPL 的 ultralytics 系）。"""
        repos = {entry.repo for entry in models.MODELS.values()}
        assert repos == {"SacredNoir/isnet-general-use-onnx", "opencv/image_segmentation_efficientsam"}


class TestDirectories:
    def test_user_dir_honours_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path))
        assert models.user_models_dir() == tmp_path

    def test_search_order_prefers_bundled(self, monkeypatch, tmp_path):
        monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path / "user"))
        dirs = models.search_dirs()
        assert dirs[0] == models.app_root() / "models"
        assert dirs[-1] == tmp_path / "user"


class TestVerify:
    def test_accepts_matching_file(self, tmp_path):
        entry = fake_entry(PAYLOAD)
        target = tmp_path / entry.filename
        target.write_bytes(PAYLOAD)
        assert models.verify(entry, target)

    def test_rejects_tampered_file(self, tmp_path):
        entry = fake_entry(PAYLOAD)
        target = tmp_path / entry.filename
        target.write_bytes(PAYLOAD + b"x")
        assert not models.verify(entry, target)

    def test_rejects_missing_file(self, tmp_path):
        assert not models.verify(fake_entry(PAYLOAD), tmp_path / "nope.onnx")


class TestDownload:
    def test_downloads_and_verifies(self, server, tmp_path):
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD

        result = models.download(entry, target_dir=tmp_path, mirrors=[server.base_url])

        assert result.read_bytes() == PAYLOAD
        assert not (tmp_path / (entry.filename + ".part")).exists()

    def test_reports_progress(self, server, tmp_path):
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD
        seen: list[tuple[str, int, int]] = []

        models.download(
            entry, target_dir=tmp_path, mirrors=[server.base_url],
            progress=lambda name, done, total: seen.append((name, done, total)),
        )

        assert seen
        assert seen[-1][1] == len(PAYLOAD)
        assert all(done <= total for _, done, total in seen)

    def test_skips_download_when_already_valid(self, server, tmp_path):
        """已装好且哈希一致时不应该再发请求（批量处理里这很关键）。"""
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD
        (tmp_path / entry.filename).write_bytes(PAYLOAD)

        models.download(entry, target_dir=tmp_path, mirrors=[server.base_url])

        assert server.hits == []

    def test_falls_back_to_next_mirror(self, server, tmp_path):
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD
        dead = "http://127.0.0.1:1"  # 必定连不上的端口

        result = models.download(entry, target_dir=tmp_path, mirrors=[dead, server.base_url])

        assert result.read_bytes() == PAYLOAD

    def test_rejects_corrupted_download(self, server, tmp_path):
        """服务端给的内容与清单里的哈希不符时必须失败，而不是装上一个坏文件。"""
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD + b"tampered"

        with pytest.raises(ModelError) as excinfo:
            models.download(entry, target_dir=tmp_path, mirrors=[server.base_url])

        assert "SHA-256" in str(excinfo.value)
        assert not (tmp_path / entry.filename).exists()

    def test_all_sources_failing_lists_each_reason(self, server, tmp_path):
        entry = fake_entry(PAYLOAD)
        server.fail_paths.add(server.path_for(entry))

        with pytest.raises(ModelError) as excinfo:
            models.download(entry, target_dir=tmp_path, mirrors=[server.base_url])

        message = str(excinfo.value)
        assert "500" in message or "boom" in message
        assert entry.filename in message  # 告诉用户手动放哪

    def test_resumes_from_partial_file(self, server, tmp_path):
        """上次中断留下的 .part 应当被续传，而不是从零开始。"""
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD
        partial = tmp_path / (entry.filename + ".part")
        partial.write_bytes(PAYLOAD[:20000])

        result = models.download(entry, target_dir=tmp_path, mirrors=[server.base_url])

        assert result.read_bytes() == PAYLOAD
        assert any(range_header for _, range_header in server.hits)

    def test_retries_and_resumes_after_truncated_response(self, server, tmp_path):
        """镜像把连接掐断时应当自动续传重试，而不是直接失败。

        这是实际踩到的坑：镜像对 44 MB 的文件会在 30 MB 附近断一次，
        而 urllib 的 read() 在断流时返回空串而不是抛异常，若不自己核对字节数，
        就会留下一个"看起来下完了"的残缺文件。
        """
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD
        server.truncate_once.add(server.path_for(entry))

        result = models.download(entry, target_dir=tmp_path, mirrors=[server.base_url])

        assert result.read_bytes() == PAYLOAD
        assert not (tmp_path / (entry.filename + ".part")).exists()

    def test_cancel_during_transfer_keeps_partial(self, server, tmp_path):
        """取消发生在下载途中时，已收到的部分要保留下来，下次可以续传。"""
        entry = fake_entry(PAYLOAD)
        server.routes[server.path_for(entry)] = PAYLOAD
        cancelled = {"flag": False}

        def on_progress(_name, done, _total):
            if done > 0:
                cancelled["flag"] = True

        with pytest.raises(ModelError):
            models.download(
                entry, target_dir=tmp_path, mirrors=[server.base_url],
                progress=on_progress, should_cancel=lambda: cancelled["flag"],
            )

        partial = tmp_path / (entry.filename + ".part")
        assert partial.exists()
        assert 0 < partial.stat().st_size < len(PAYLOAD)

        # 已经收到的部分保留下来，下次可以接着下
        assert (tmp_path / (entry.filename + ".part")).exists()


class TestImportArchive:
    def _bundle(self, path: Path, payload: bytes, filename: str) -> Path:
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr(f"some/nested/{filename}", payload)
        return path

    def _register_fake(self, monkeypatch, payload: bytes) -> ModelEntry:
        """把一个内容已知的假模型登记进清单，用来验证导入的成功路径。"""
        entry = ModelEntry(
            key="fake-bundle",
            label="打包测试模型",
            filename="fake-bundle.onnx",
            repo="test/repo",
            sha256=hashlib.sha256(payload).hexdigest(),
            purpose="测试用",
        )
        monkeypatch.setitem(models.MODELS, entry.key, entry)
        return entry

    def test_imports_valid_bundle(self, monkeypatch, tmp_path):
        """文件放在压缩包的任意层级都应被认出，且导入后哈希一致。"""
        payload = b"real-model-bytes"
        entry = self._register_fake(monkeypatch, payload)
        archive = self._bundle(tmp_path / "bundle.zip", payload, entry.filename)

        imported = models.import_archive(archive, target_dir=tmp_path / "models")

        assert imported == [entry.key]
        target = tmp_path / "models" / entry.filename
        assert target.read_bytes() == payload
        assert models.verify(entry, target)

    def test_rejects_corrupted_bundle_and_cleans_up(self, monkeypatch, tmp_path):
        """校验不过必须删掉半成品，不能留下一个"看起来装好了"的坏文件。"""
        entry = self._register_fake(monkeypatch, b"expected-bytes")
        archive = self._bundle(tmp_path / "bundle.zip", b"tampered-bytes", entry.filename)

        with pytest.raises(ModelError, match="校验失败"):
            models.import_archive(archive, target_dir=tmp_path / "models")

        assert not (tmp_path / "models" / entry.filename).exists()

    def test_rejects_archive_without_known_models(self, tmp_path):
        archive = self._bundle(tmp_path / "bundle.zip", b"x", "something_else.onnx")

        with pytest.raises(ModelError, match="没有找到已知的模型文件"):
            models.import_archive(archive, target_dir=tmp_path / "models")

    def test_missing_archive_is_reported(self, tmp_path):
        with pytest.raises(ModelError, match="找不到压缩包"):
            models.import_archive(tmp_path / "nope.zip", target_dir=tmp_path / "models")


class TestStatus:
    def test_reports_missing_when_nothing_installed(self, monkeypatch, tmp_path):
        monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path / "empty"))
        monkeypatch.setattr(models, "app_root", lambda: tmp_path / "no_app")
        assert len(models.missing()) == len(models.MODELS)

    def test_finds_model_in_user_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path))
        monkeypatch.setattr(models, "app_root", lambda: tmp_path / "no_app")
        entry = models.MODELS["isnet-general-use"]
        (tmp_path / entry.filename).write_bytes(b"x")

        assert models.is_installed(entry)
        assert entry not in models.missing()
        assert "isnet-general-use" in models.describe_status()

    def test_bundled_dir_wins_over_user_dir(self, monkeypatch, tmp_path):
        bundled = tmp_path / "app" / "models"
        bundled.mkdir(parents=True)
        user = tmp_path / "user"
        user.mkdir()
        entry = models.MODELS["efficientsam-ti"]
        (bundled / entry.filename).write_bytes(b"bundled")
        (user / entry.filename).write_bytes(b"user")

        monkeypatch.setattr(models, "app_root", lambda: tmp_path / "app")
        monkeypatch.setenv(models.ENV_MODELS_DIR, str(user))

        assert models.find(entry) == bundled / entry.filename
