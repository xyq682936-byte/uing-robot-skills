"""Synthetic-only tests. No Isaac process, external server, or user grasp files."""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "isaac-grasp-direct-control/scripts/success_upload.py"
spec = importlib.util.spec_from_file_location("success_upload", SCRIPT)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.export = self.root / "export"
        self.export.mkdir()
        self.box = self.root / "outbox"
        self.manifest_path = self.export / "export.json"
        self.roles = sorted(upload.REQUIRED_ROLES)
        for role in self.roles:
            (self.export / (role + ".json")).write_text(json.dumps({"synthetic": True, "role": role}), encoding="utf-8")
        self.manifest = {"schema_version": 2, **{k: k + "-1" for k in
            ("record_id", "record_revision", "object_id", "batch_id", "strategy_id", "strategy_revision", "trial_id", "arm_id")},
            "verification": {"review_state": "reviewed", "grasp_success": True, "scope": "exploration",
                             "reviewer": "synthetic-test", "reviewed_at": "2026-10-22T12:00:00Z", "criteria_ref": "configuration.json"},
            **{k: {"synthetic": True} for k in upload.GROUPS}, "omissions": [],
            "files": [{"path": role + ".json", "role": role} for role in self.roles]}
        self.manifest["standard"].update(status="not_tested", reference="synthetic-project-test", version="1",
                                         applicable_clauses=[], adaptations=[], threshold_sources=["configuration.json"])
        for group in ("object", "robot", "control", "environment"):
            self.manifest[group]["effective_parameters_ref"] = "configuration.json"
        identity = {"manufacturer": "Synthetic lab", "model": "Test model", "model_version": "test-1",
                    "asset_sha256": hashlib.sha256(b"synthetic asset identity").hexdigest()}
        self.manifest["robot"].update(identity, gripper=dict(identity))
        self.manifest["provenance"].update(code_revision="synthetic-revision", units_and_frames_ref="configuration.json")
        self.config_path = self.root / "config.json"
        self.config = {"schema_version": 1, "enabled": True, "endpoint": "https://receiver.example.invalid/upload",
                       "consent": {"granted": True, "destination": "https://receiver.example.invalid/upload",
                                   "scope": upload.SCOPE, "recorded_at": "2026-10-22T12:00:00Z"},
                       "token_env": "ISAAC_UPLOAD_TEST_TOKEN", "timeout_seconds": 2,
                       "max_attempts": 3, "retry_delay_seconds": 1}
        env = patch.dict(os.environ, {"ISAAC_UPLOAD_TEST_TOKEN": "synthetic-token"})
        env.start()
        self.addCleanup(env.stop)
        self.save()

    def save(self):
        upload.atomic_json(self.manifest_path, self.manifest)
        upload.atomic_json(self.config_path, self.config)

    def pack(self):
        self.save()
        return upload.pack(self.export, self.manifest_path, self.box)["package_id"]

    def job(self, pid):
        return upload.read_json(self.box / "jobs" / (pid + ".json"))

    def change_job(self, pid, **changes):
        job = self.job(pid)
        job.update(changes)
        upload.atomic_json(self.box / "jobs" / (pid + ".json"), job)

    @staticmethod
    def good_response(endpoint, token, stream, size, pid, timeout):
        data = stream.read()
        assert len(data) == size
        assert hashlib.sha256(data).hexdigest() == pid
        return 201, upload.canonical({"stored": True, "package_id": pid, "sha256": pid})

    def test_package_manifest_hashes_and_dedup_after_upload(self):
        pid = self.pack()
        archive_path = self.box / "packages" / (pid + ".zip")
        self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), pid)
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["standard"]["status"], "not_tested")
            for entry in manifest["files"]:
                content = archive.read(entry["path"])
                self.assertEqual(len(content), entry["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
        with patch.object(upload, "post_bundle", side_effect=self.good_response) as transport:
            self.assertEqual(upload.send(self.config_path, self.box)[0]["state"], "uploaded")
            self.assertEqual(self.pack(), pid)
            self.assertEqual(upload.send(self.config_path, self.box), [])
            self.assertEqual(transport.call_count, 1)
        self.assertEqual(self.job(pid)["total_attempts"], 1)
        self.assertEqual(len(list((self.box / "jobs").glob("*.json"))), 1)

    def test_robot_and_gripper_identity_required(self):
        original = copy.deepcopy(self.manifest)
        for group in ("robot", "gripper"):
            for field in ("manufacturer", "model", "model_version", "asset_sha256"):
                self.manifest = copy.deepcopy(original)
                target = self.manifest["robot"] if group == "robot" else self.manifest["robot"]["gripper"]
                del target[field]
                with self.subTest(group=group, field=field), self.assertRaises(upload.UploadError):
                    self.pack()

    def test_missing_or_wrong_role_reference_rejected(self):
        for reference in ("missing-config.json", "actions.json"):
            self.manifest["robot"]["effective_parameters_ref"] = reference
            with self.subTest(reference=reference), self.assertRaises(upload.UploadError):
                self.pack()

    def test_unrated_measurement_can_keep_grasp_success(self):
        self.manifest["standard"]["status"] = "unrated"
        self.assertEqual(self.job(self.pack())["state"], "queued")
        self.manifest["verification"]["scope"] = "standard"
        with self.assertRaises(upload.UploadError):
            self.pack()

    def test_invalid_queue_job_does_not_hide_valid_job(self):
        self.pack()
        bad = self.box / "jobs/000-invalid.json"
        bad.write_text('{broken', encoding="utf-8")
        with patch.object(upload, "post_bundle", side_effect=self.good_response):
            states = [item["state"] for item in upload.send(self.config_path, self.box)]
        self.assertIn("invalid", states)
        self.assertIn("uploaded", states)
        self.assertEqual(upload.status(self.box)["counts"]["invalid"], 1)
        self.assertEqual(bad.read_text(), '{broken')

    def rewrite_package(self, mutate):
        pid = self.pack()
        with zipfile.ZipFile(self.box / 'packages' / (pid + '.zip')) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(members['manifest.json'])
        mutate(manifest, members)
        members['manifest.json'] = upload.canonical(manifest)
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as archive:
            for name, content in members.items():
                archive.writestr(upload.zip_info(name), content)
        payload = output.getvalue()
        new_pid = hashlib.sha256(payload).hexdigest()
        (self.box / 'packages' / (new_pid + '.zip')).write_bytes(payload)
        job = self.job(pid)
        job['package_id'] = new_pid
        upload.atomic_json(self.box / 'jobs' / (new_pid + '.json'), job)
        if new_pid != pid:
            (self.box / 'jobs' / (pid + '.json')).unlink()
        return new_pid

    def test_legacy_manifest_is_retained_and_blocked(self):
        self.manifest['schema_version'] = 1
        with self.assertRaisesRegex(upload.UploadError, 'unsupported_manifest_schema'):
            self.pack()
        self.assertTrue(self.manifest_path.is_file())
        self.manifest['schema_version'] = 2
        pid = self.rewrite_package(lambda m, _: m.update(schema_version=1))
        with patch.object(upload, 'post_bundle') as transport:
            self.assertEqual(upload.send(self.config_path, self.box)[0]['state'], 'needs_review')
            transport.assert_not_called()
        self.assertEqual(self.job(pid)['attempts'], 0)
        self.assertTrue((self.box / 'packages' / (pid + '.zip')).is_file())
        with self.assertRaises(upload.UploadError):
            upload.retry(self.box, pid)

    def test_packaged_references_resolve_under_declared_data_root(self):
        pid = self.pack()
        with (self.box / 'packages' / (pid + '.zip')).open('rb') as stream:
            manifest = upload.validate_package(stream, self.job(pid))
        with zipfile.ZipFile(self.box / 'packages' / (pid + '.zip')) as archive:
            self.assertEqual(manifest['data_root'], 'data')
            for group in ('object', 'robot', 'control', 'environment'):
                path = manifest['data_root'] + '/' + manifest[group]['effective_parameters_ref']
                self.assertTrue(json.loads(archive.read(path))['synthetic'])

    def test_metadata_shape_and_timestamp_validation(self):
        original = copy.deepcopy(self.manifest)
        mutations = [lambda m: m['standard'].update(version=''),
                     lambda m: m['standard'].update(applicable_clauses='all'),
                     lambda m: m['standard'].update(adaptations=None),
                     lambda m: m['standard'].update(threshold_sources=[]),
                     lambda m: m['verification'].update(reviewed_at='yesterday'),
                     lambda m: m['verification'].update(reviewed_at='2026-10-22T12:00:00'),
                     lambda m: m['robot'].update(model='unknown'),
                     lambda m: m['robot'].update(asset_sha256='wrong'),
                     lambda m: m['provenance'].update(code_revision=''),
                     lambda m: m.update(schema_version=True)]
        for index, mutate in enumerate(mutations):
            self.manifest = copy.deepcopy(original)
            mutate(self.manifest)
            with self.subTest(index=index), self.assertRaises(upload.UploadError):
                self.pack()

    def test_all_configuration_references_are_checked(self):
        original = copy.deepcopy(self.manifest)
        mutations = [lambda m: m['object'].pop('effective_parameters_ref'),
                     lambda m: m['control'].update(effective_parameters_ref='missing.json'),
                     lambda m: m['environment'].update(effective_parameters_ref='trace.json'),
                     lambda m: m['verification'].update(criteria_ref='missing.json'),
                     lambda m: m['provenance'].update(units_and_frames_ref='actions.json'),
                     lambda m: m['standard'].update(threshold_sources=['missing.json'])]
        for index, mutate in enumerate(mutations):
            self.manifest = copy.deepcopy(original)
            mutate(self.manifest)
            with self.subTest(index=index), self.assertRaises(upload.UploadError):
                self.pack()

    def test_configuration_content_missing_or_sensitive_rejected(self):
        config = self.export / 'configuration.json'
        for payload in (b'{}', b'[]', b'not json', b'{"x":NaN}', b'{"api_key":"test-only"}',
                        b'{"path":"C:/Users/synthetic/private"}'):
            config.write_bytes(payload)
            with self.subTest(payload=payload), self.assertRaises(upload.UploadError):
                self.pack()

    def test_archived_hash_or_identity_mismatch_never_sent(self):
        pid = self.rewrite_package(lambda m, files: files.update({'data/actions.json': b'{"changed": true}'}))
        with patch.object(upload, 'post_bundle') as transport:
            self.assertEqual(upload.send(self.config_path, self.box)[0]['state'], 'needs_review')
            transport.assert_not_called()
        self.assertEqual(self.job(pid)['attempts'], 0)
        pid = self.pack()
        self.change_job(pid, record_id='wrong-record')
        with patch.object(upload, 'post_bundle') as transport:
            self.assertEqual(upload.send(self.config_path, self.box)[0]['state'], 'needs_review')
            transport.assert_not_called()

    def test_revocation_during_package_validation_prevents_post(self):
        self.pack()
        original = upload.validate_package
        def revoke(*args):
            result = original(*args)
            self.config['enabled'] = False
            self.save()
            return result
        with patch.object(upload, 'validate_package', side_effect=revoke), patch.object(upload, 'post_bundle') as transport:
            with self.assertRaisesRegex(upload.UploadError, 'disabled'):
                upload.send(self.config_path, self.box)
            transport.assert_not_called()

    def test_damaged_job_counters_do_not_block_other_records(self):
        pid = self.pack()
        self.change_job(pid, total_attempts=None)
        self.manifest['record_id'] = 'second-record'
        self.pack()
        with patch.object(upload, 'post_bundle', side_effect=self.good_response) as transport:
            states = [x['state'] for x in upload.send(self.config_path, self.box)]
            self.assertIn('invalid', states)
            self.assertIn('uploaded', states)
            self.assertEqual(transport.call_count, 1)

    def test_invalid_jobs_do_not_consume_send_limit(self):
        self.pack()
        for i in range(6):
            (self.box / 'jobs' / f'000-invalid-{i}.json').write_text('{}')
        with patch.object(upload, 'post_bundle', side_effect=self.good_response) as transport:
            result = upload.send(self.config_path, self.box, limit=1)
            self.assertEqual(sum(x['state'] == 'uploaded' for x in result), 1)
            self.assertEqual(transport.call_count, 1)

    def test_configuration_mutation_between_validation_and_copy_is_rejected(self):
        original = upload.validate_manifest
        def change_after_validation(*args):
            result = original(*args)
            (self.export / 'configuration.json').write_text('{}')
            return result
        with patch.object(upload, 'validate_manifest', side_effect=change_after_validation):
            with self.assertRaises(upload.UploadError):
                self.pack()
        self.assertEqual(list((self.box / 'jobs').glob('*.json')), [])

    def test_formal_pass_requires_clauses_and_keeps_scope(self):
        self.manifest['standard']['status'] = 'passed'
        self.manifest['verification']['scope'] = 'standard'
        with self.assertRaises(upload.UploadError):
            self.pack()
        self.manifest['standard']['applicable_clauses'] = ['synthetic-project-condition-1']
        self.assertEqual(self.job(self.pack())['state'], 'queued')

    def test_pending_failed_and_false_standard_pass_rejected(self):
        cases = [{"review_state": "pending"}, {"grasp_success": False}, {"scope": "standard"}, {"reviewer": ""}]
        original = copy.deepcopy(self.manifest)
        for change in cases:
            with self.subTest(change=change):
                self.manifest = copy.deepcopy(original)
                self.manifest["verification"].update(change)
                with self.assertRaises(upload.UploadError):
                    self.pack()

    def test_missing_empty_and_unavailable_evidence(self):
        original = copy.deepcopy(self.manifest)
        for role in self.roles:
            with self.subTest(role=role):
                self.manifest = copy.deepcopy(original)
                self.manifest["files"] = [e for e in self.manifest["files"] if e["role"] != role]
                with self.assertRaises(upload.UploadError):
                    self.pack()
        self.manifest = original
        (self.export / "trace.json").write_bytes(b"")
        with self.assertRaisesRegex(upload.UploadError, "empty_evidence"):
            self.pack()

    def test_portable_path_validation(self):
        for name in ("../secret.txt", "/etc/file", "C:/file", "a\\b", "x:stream", "a//b", "./x", ".env", "id_rsa", "private.key"):
            with self.subTest(name=name), self.assertRaises(upload.UploadError):
                upload.source_path(self.export, name)

    def test_link_escape_rejected(self):
        external_dir = self.root / "outside"
        external_dir.mkdir()
        external = external_dir / "outside.txt"
        external.write_text("synthetic external file", encoding="utf-8")
        link = self.export / "escape.txt"
        try:
            link.symlink_to(external)
        except OSError:
            if os.name != "nt":
                self.skipTest("OS does not allow creating test symlinks")
            junction = self.export / "junction"
            quoted_link = str(junction).replace("'", "''")
            quoted_target = str(external_dir).replace("'", "''")
            result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                                     f"$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path '{quoted_link}' -Target '{quoted_target}' | Out-Null"],
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, "Cannot create Windows test junction")
            with self.assertRaises(upload.UploadError):
                upload.source_path(self.export, "junction/outside.txt")
            return
        with self.assertRaises(upload.UploadError):
            upload.source_path(self.export, "escape.txt")

    def test_credentials_and_absolute_paths_rejected_in_metadata(self):
        for value in ({"api_key": "synthetic"}, {"path": "C:\\Users\\someone\\data"},
                      {"path": "/home/someone/data"}, {"value": "-----BEGIN PRIVATE KEY-----"}):
            with self.subTest(value=value), self.assertRaises(upload.UploadError):
                upload.check_metadata(value)

    def test_disabled_missing_consent_destination_and_token_never_connect(self):
        self.pack()
        original = copy.deepcopy(self.config)
        for scenario in ("disabled", "no_endpoint", "consent", "destination", "scope", "token", "http", "query"):
            self.config = copy.deepcopy(original)
            if scenario == "disabled": self.config["enabled"] = False
            if scenario == "no_endpoint": self.config["endpoint"] = ""
            if scenario == "consent": self.config["consent"]["granted"] = False
            if scenario == "destination": self.config["consent"]["destination"] = "https://other.invalid"
            if scenario == "scope": self.config["consent"]["scope"] = "whole_computer"
            if scenario == "token": self.config["token_env"] = "MISSING_SYNTHETIC_TOKEN_5991"
            if scenario == "http": self.config["endpoint"] = "http://receiver.invalid/upload"
            if scenario == "query": self.config["endpoint"] += "?token=synthetic"
            self.save()
            with self.subTest(scenario=scenario), patch.object(upload, "post_bundle") as transport:
                with self.assertRaises(upload.UploadError):
                    upload.send(self.config_path, self.box)
                transport.assert_not_called()

    def test_tamper_blocks_transport(self):
        pid = self.pack()
        (self.box / "packages" / (pid + ".zip")).write_bytes(b"tampered synthetic bytes")
        with patch.object(upload, "post_bundle") as transport:
            self.assertEqual(upload.send(self.config_path, self.box)[0]["state"], "corrupt")
            transport.assert_not_called()
        self.assertEqual(self.job(pid)["attempts"], 0)

    def test_invalid_ack_not_uploaded(self):
        self.pack()
        with patch.object(upload, "post_bundle", return_value=(200, b'{"stored":true,"sha256":"wrong"}')):
            self.assertEqual(upload.send(self.config_path, self.box)[0]["state"], "rejected")

    def test_retries_bounded_backoff_and_explicit_restart(self):
        pid = self.pack()
        with patch.object(upload, "post_bundle", side_effect=TimeoutError("synthetic timeout")) as transport:
            for attempt in range(1, 4):
                result = upload.send(self.config_path, self.box)
                self.assertEqual(result[0]["state"], "exhausted" if attempt == 3 else "retry_wait")
                self.assertEqual(upload.send(self.config_path, self.box), [])
                self.change_job(pid, next_attempt_at=0)
            self.assertEqual(transport.call_count, 3)
            upload.retry(self.box, pid)
            self.assertEqual(self.job(pid)["total_attempts"], 3)
            self.assertEqual(self.job(pid)["attempts"], 0)

    def test_auth_failure_stops_batch(self):
        self.pack()
        self.manifest["trial_id"] = "trial-2"
        self.manifest["record_id"] = "record-2"
        self.pack()
        with patch.object(upload, "post_bundle", return_value=(401, b"synthetic response")) as transport:
            self.assertEqual(upload.send(self.config_path, self.box)[0]["state"], "auth_blocked")
            self.assertEqual(transport.call_count, 1)

    def test_redirect_never_followed(self):
        self.pack()
        with patch.object(upload, "post_bundle", return_value=(307, b"")) as transport:
            self.assertEqual(upload.send(self.config_path, self.box)[0]["state"], "rejected")
            self.assertEqual(transport.call_count, 1)

    def test_config_revocation_stops_next_job(self):
        self.pack()
        self.manifest["record_id"] = "record-2"
        self.pack()
        def revoke(*args):
            self.config["enabled"] = False
            self.save()
            return self.good_response(*args)
        with patch.object(upload, "post_bundle", side_effect=revoke) as transport:
            with self.assertRaisesRegex(upload.UploadError, "disabled"):
                upload.send(self.config_path, self.box)
            self.assertEqual(transport.call_count, 1)

    def test_changed_destination_cannot_resend_attempted_job(self):
        self.pack()
        with patch.object(upload, "post_bundle", return_value=(503, b"")):
            upload.send(self.config_path, self.box)
        pid = upload.status(self.box)["jobs"][0]["package_id"]
        self.change_job(pid, next_attempt_at=0)
        self.config["endpoint"] = "https://different.example.invalid/upload"
        self.config["consent"]["destination"] = self.config["endpoint"]
        self.save()
        with patch.object(upload, "post_bundle") as transport:
            with self.assertRaisesRegex(upload.UploadError, "destination_changed"):
                upload.send(self.config_path, self.box)
            transport.assert_not_called()

    def test_worker_lock_and_crash_release(self):
        self.pack()
        code = "import importlib.util,sys; s=importlib.util.spec_from_file_location('u',sys.argv[1]); u=importlib.util.module_from_spec(s); s.loader.exec_module(u); l=u.queue_lock(sys.argv[2]); l.__enter__()"
        with upload.queue_lock(self.box):
            child = subprocess.run([sys.executable, "-B", "-c", code, str(SCRIPT), str(self.box)], capture_output=True, timeout=10)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn(b"outbox_busy", child.stderr)
        child = subprocess.run([sys.executable, "-B", "-c", code + "; import os; os._exit(0)", str(SCRIPT), str(self.box)], capture_output=True, timeout=10)
        self.assertEqual(child.returncode, 0)
        with upload.queue_lock(self.box):
            pass

    def test_real_loopback_streaming_transport_synthetic_only(self):
        pid = self.pack()
        captured = {}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                captured.update(path=self.path, headers=dict(self.headers), sha256=hashlib.sha256(payload).hexdigest())
                response = upload.canonical({"stored": True, "package_id": captured["sha256"], "sha256": captured["sha256"]})
                self.send_response(201)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.config["endpoint"] = f"https://127.0.0.1:{server.server_port}/upload"
        self.config["consent"]["destination"] = self.config["endpoint"]
        self.save()
        def loopback_http(host, port, timeout, context):
            self.assertEqual(host, "127.0.0.1")
            return http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            # Production HTTPS validation remains intact; only this test substitutes the connection.
            with patch.object(upload.http.client, "HTTPSConnection", side_effect=loopback_http):
                result = upload.send(self.config_path, self.box)
            self.assertEqual(result[0]["state"], "uploaded")
            self.assertEqual(captured["sha256"], pid)
            self.assertEqual(captured["headers"]["Idempotency-Key"], pid)
            self.assertEqual(captured["headers"]["Authorization"], "Bearer synthetic-token")
            self.assertEqual(captured["path"], "/upload")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
