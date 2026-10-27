#!/usr/bin/env python3
"""Pack reviewed grasp exports; optionally upload to an explicitly configured receiver.

Python 3.10+, standard library only. No Isaac dependency, auto-start or disk discovery.
See references/success-data-upload.md for the export and receiver contracts.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import ssl
import time
from urllib.parse import urlsplit
import uuid
import zipfile

SCOPE = "verified_success_full_bundle_v1"
MANIFEST_VERSION = 2
MAX_METADATA_BYTES = 4 * 1024 * 1024
ROLES = {"actions", "observations", "verification", "trace", "configuration",
         "asset", "media", "standard", "calibration"}
REQUIRED_ROLES = {"actions", "observations", "verification", "trace", "configuration"}
GROUPS = ("object", "robot", "control", "environment", "standard", "provenance")
STATES = {"queued", "retry_wait", "uploaded", "rejected", "auth_blocked",
          "corrupt", "exhausted", "needs_review"}
SECRET_KEYS = {"password", "passwd", "secret", "client_secret", "api_key", "apikey",
               "access_token", "refresh_token", "authorization", "private_key"}
CHUNK = 1024 * 1024


class UploadError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise UploadError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def queue_lock(outbox):
    """OS releases this lock on process exit, including a crash."""
    outbox = Path(outbox).resolve()
    outbox.mkdir(parents=True, exist_ok=True)
    with (outbox / ".worker.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise UploadError("outbox_busy: try again after the current worker finishes") from exc
        try:
            (outbox / "packages").mkdir(exist_ok=True)
            (outbox / "jobs").mkdir(exist_ok=True)
            yield outbox
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def check_metadata(value):
    """Basic export hygiene only; this is not a general-purpose secret detector."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower().replace("-", "_") not in SECRET_KEYS,
                    "credential_field_in_manifest")
            check_metadata(child)
    elif isinstance(value, list):
        for child in value:
            check_metadata(child)
    elif isinstance(value, str):
        require("PRIVATE KEY-----" not in value, "private_key_in_manifest")
        require(not re.match(r"^(?:[A-Za-z]:[\\/]|/|\\\\)", value),
                "absolute_local_path_in_manifest: use portable relative references")


def portable_name(name):
    require(isinstance(name, str) and name, "file_path_required")
    parts = name.split("/")
    require(not any(p in ("", ".", "..") for p in parts)
            and not any(c in name for c in ("\\", ":", "\x00")), "unsafe_file_path")
    require(not any(p.lower() in {".env", ".git", "credentials", "credentials.json",
                                 "id_rsa", "id_ed25519"}
                    or p.lower().startswith(".env.")
                    or p.lower().endswith((".pem", ".key", ".pfx", ".p12"))
                    for p in parts), "credential_like_file_not_allowed")
    return parts


def source_path(root, name):
    parts = portable_name(name)
    candidate = root.joinpath(*parts)
    # Reject links/reparse directories as well as paths escaping the export root.
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        info = parent.lstat()
        require(not parent.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400),
                "linked_export_file_not_allowed")
    resolved = candidate.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), "export_file_outside_root")
    require(resolved.stat().st_size > 0, "empty_evidence_file")
    return resolved


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def timestamp(value, label):
    require(nonblank(value), "missing_" + label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "timezone_required_" + label)
    except ValueError as exc:
        raise UploadError("invalid_" + label) from exc


def validate_contract(manifest, file_roles):
    require(isinstance(manifest, dict) and type(manifest.get("schema_version")) is int
            and manifest["schema_version"] == MANIFEST_VERSION,
            "unsupported_manifest_schema")
    for key in ("record_id", "record_revision", "object_id", "batch_id", "strategy_id",
                "strategy_revision", "trial_id", "arm_id"):
        require(nonblank(manifest.get(key)), "missing_" + key)
    review = manifest.get("verification", {})
    require(isinstance(review, dict) and review.get("review_state") == "reviewed"
            and review.get("grasp_success") is True, "only_reviewed_grasp_success_is_eligible")
    require(review.get("scope") in ("exploration", "repeatability", "standard"), "invalid_success_scope")
    for key in ("reviewer", "reviewed_at", "criteria_ref"):
        require(nonblank(review.get(key)), "missing_review_" + key)
    timestamp(review["reviewed_at"], "reviewed_at")
    for key in GROUPS:
        require(isinstance(manifest.get(key), dict) and manifest[key], "missing_metadata_" + key)
    robot = manifest["robot"]
    for label, identity in (("robot", robot), ("gripper", robot.get("gripper"))):
        require(isinstance(identity, dict), "missing_" + label + "_identity")
        for field in ("manufacturer", "model", "model_version"):
            value = identity.get(field)
            require(nonblank(value) and value.strip().casefold() not in
                    {"unknown", "n/a", "none", "null", "tbd", "未知", "待填写", "待补全"},
                    "missing_" + label + "_" + field)
        require(isinstance(identity.get("asset_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", identity["asset_sha256"]), "invalid_" + label + "_asset_sha256")
    standard = manifest["standard"]
    require(standard.get("status") in ("passed", "failed", "not_tested", "incomplete", "not_applicable", "unrated"),
            "standard_status_required")
    for key in ("reference", "version"):
        require(nonblank(standard.get(key)), "missing_standard_" + key)
    for key in ("applicable_clauses", "adaptations", "threshold_sources"):
        require(isinstance(standard.get(key), list), "invalid_standard_" + key)
    require(all(nonblank(item) for item in standard["applicable_clauses"]), "invalid_standard_applicable_clauses")
    require(bool(standard["threshold_sources"]), "missing_standard_threshold_sources")
    if standard["status"] == "passed":
        require(bool(standard["applicable_clauses"]), "passed_standard_requires_applicable_clauses")
    require(review["scope"] != "standard" or standard["status"] == "passed",
            "standard_scope_requires_passed_standard")
    require(isinstance(manifest.get("omissions"), list), "omissions_list_required")
    for omission in manifest["omissions"]:
        require(isinstance(omission, dict) and nonblank(omission.get("field")) and nonblank(omission.get("reason")),
                "omission_requires_field_and_reason")
    require(nonblank(manifest["provenance"].get("code_revision")), "missing_code_revision")
    references = [review["criteria_ref"], manifest["provenance"].get("units_and_frames_ref")]
    references += [manifest[group].get("effective_parameters_ref") for group in ("object", "robot", "control", "environment")]
    references += standard["threshold_sources"]
    for reference in references:
        portable_name(reference)
        require(file_roles.get(reference) == "configuration", "configuration_reference_missing_or_wrong_role")
    check_metadata(manifest)
    canonical(manifest)
    return set(references)


def metadata_json(payload):
    require(len(payload) <= MAX_METADATA_BYTES, "metadata_json_too_large")
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise UploadError("invalid_metadata_json") from exc
    require(isinstance(value, dict) and value, "empty_or_invalid_metadata_object")
    check_metadata(value)
    try:
        canonical(value)
    except (TypeError, ValueError) as exc:
        raise UploadError("invalid_metadata_value") from exc
    return value


def validate_manifest(manifest, root):
    require(isinstance(manifest, dict), "invalid_manifest")
    files = manifest.get("files")
    require(isinstance(files, list) and files, "explicit_file_list_required")
    entries, names, roles, file_roles = [], set(), set(), {}
    for entry in files:
        require(isinstance(entry, dict) and isinstance(entry.get("role"), str) and entry["role"] in ROLES, "invalid_file_role")
        name = entry.get("path")
        path = source_path(root, name)
        require(name.casefold() not in names, "duplicate_file_path")
        names.add(name.casefold())
        roles.add(entry["role"])
        file_roles[name] = entry["role"]
        entries.append((name, entry["role"], path))
    require(REQUIRED_ROLES <= roles, "missing_required_evidence_roles")
    references = validate_contract(manifest, file_roles)
    require(len(canonical(manifest)) <= MAX_METADATA_BYTES, "metadata_json_too_large")
    for reference in references:
        with source_path(root, reference).open("rb") as stream:
            metadata_json(stream.read(MAX_METADATA_BYTES + 1))
    return sorted(entries)


def validate_package(stream, job=None):
    """Check the archived contract as well as referenced evidence; never extract files."""
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and "manifest.json" in names, "invalid_archive_members")
        require(archive.getinfo("manifest.json").file_size <= MAX_METADATA_BYTES, "metadata_json_too_large")
        manifest = metadata_json(archive.read("manifest.json"))
        require(manifest.get("data_root") == "data", "invalid_archive_data_root")
        if job is not None:
            require(all(manifest.get(key) == job.get(key) for key in ("record_id", "record_revision")), "job_manifest_identity_mismatch")
        entries = manifest.get("files")
        require(isinstance(entries, list) and entries, "explicit_file_list_required")
        roles, file_roles, expected = set(), {}, {"manifest.json"}
        case_names = set()
        for entry in entries:
            require(isinstance(entry, dict) and isinstance(entry.get("role"), str) and entry["role"] in ROLES, "invalid_file_role")
            name = entry.get("path")
            portable_name(name)
            require(name.startswith("data/") and name.casefold() not in case_names, "invalid_archive_file_path")
            case_names.add(name.casefold())
            require(name in names, "missing_archive_evidence")
            require(type(entry.get("bytes")) is int and entry["bytes"] > 0, "invalid_archive_file_size")
            require(archive.getinfo(name).file_size == entry["bytes"], "archive_file_size_mismatch")
            with archive.open(name) as source:
                require(digest_stream(source) == entry.get("sha256"), "archive_file_hash_mismatch")
            file_roles[name[5:]] = entry["role"]
            roles.add(entry["role"])
            expected.add(name)
        require(set(names) == expected and REQUIRED_ROLES <= roles, "unexpected_or_missing_archive_evidence")
        references = validate_contract(manifest, file_roles)
        for reference in references:
            name = "data/" + reference
            require(archive.getinfo(name).file_size <= MAX_METADATA_BYTES, "metadata_json_too_large")
            metadata_json(archive.read(name))
    stream.seek(0)
    return manifest


def zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    return info


def digest_stream(stream):
    digest = hashlib.sha256()
    while block := stream.read(CHUNK):
        digest.update(block)
    return digest.hexdigest()


def pack(export_root, manifest_path, outbox):
    root = Path(export_root).resolve(strict=True)
    manifest = read_json(manifest_path)
    entries = validate_manifest(manifest, root)
    with queue_lock(outbox) as box:
        temporary = box / "packages" / (uuid.uuid4().hex + ".tmp")
        try:
            records = []
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for name, role, path in entries:
                    before = path.stat()
                    digest, size = hashlib.sha256(), 0
                    with path.open("rb") as source, archive.open(zip_info("data/" + name), "w", force_zip64=True) as target:
                        while block := source.read(CHUNK):
                            target.write(block)
                            digest.update(block)
                            size += len(block)
                    after = path.stat()
                    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                            and size == before.st_size, "export_changed_during_pack: freeze export first")
                    records.append({"path": "data/" + name, "role": role, "bytes": size,
                                    "sha256": digest.hexdigest()})
                manifest["files"] = records
                manifest["data_root"] = "data"
                archive.writestr(zip_info("manifest.json"), canonical(manifest))
            with temporary.open("rb") as stream:
                validate_package(stream)
            with temporary.open("rb") as stream:
                package_id = digest_stream(stream)
            package = box / "packages" / (package_id + ".zip")
            if package.exists():
                with package.open("rb") as stream:
                    require(digest_stream(stream) == package_id, "existing_package_corrupt")
            else:
                os.replace(temporary, package)
            job_path = box / "jobs" / (package_id + ".json")
            if not job_path.exists():
                atomic_json(job_path, {"schema_version": 1, "package_id": package_id,
                                      "record_id": manifest["record_id"], "record_revision": manifest["record_revision"],
                                      "state": "queued", "attempts": 0, "total_attempts": 0,
                                      "created_at": time.time(), "next_attempt_at": 0,
                                      "endpoint": None, "history": []})
            return {"package_id": package_id, "state": read_json(job_path)["state"]}
        finally:
            temporary.unlink(missing_ok=True)


def upload_config(path):
    config = read_json(path)
    require(isinstance(config, dict) and type(config.get("schema_version")) is int and config["schema_version"] == 1, "unsupported_config_schema")
    require(config.get("enabled") is True, "upload_disabled")
    endpoint = config.get("endpoint", "")
    require(isinstance(endpoint, str), "invalid_endpoint")
    parsed = urlsplit(endpoint)
    require(parsed.scheme == "https" and parsed.hostname and not parsed.username
            and not parsed.password and not parsed.fragment and not parsed.query,
            "endpoint_requires_https_without_credentials_query_or_fragment")
    require(not any(ord(c) <= 32 or ord(c) >= 127 for c in endpoint), "endpoint_requires_encoded_url")
    parsed.port  # Validate invalid/out-of-range ports before attempting transport.
    consent = config.get("consent", {})
    require(isinstance(consent, dict) and consent.get("granted") is True
            and consent.get("destination") == endpoint and consent.get("scope") == SCOPE
            and bool(consent.get("recorded_at")), "explicit_destination_bound_consent_required")
    timestamp(consent["recorded_at"], "consent_recorded_at")
    for key, maximum in (("timeout_seconds", 300), ("max_attempts", 10), ("retry_delay_seconds", 86400)):
        value = config.get(key)
        require(type(value) is int and 1 <= value <= maximum, "invalid_" + key)
    env_name = config.get("token_env", "")
    require(isinstance(env_name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name), "invalid_token_env")
    token = os.environ.get(env_name, "")
    require(bool(token) and all(32 < ord(c) < 127 for c in token), "missing_or_invalid_upload_token")
    return config, token


def post_bundle(endpoint, token, stream, size, package_id, timeout):
    parsed = urlsplit(endpoint)
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443,
                                              timeout=timeout, context=ssl.create_default_context())
    try:
        connection.request("POST", parsed.path or "/", body=stream, headers={
            "Authorization": "Bearer " + token, "Content-Type": "application/zip",
            "Content-Length": str(size), "Idempotency-Key": package_id,
            "X-Content-SHA256": package_id})
        response = connection.getresponse()
        # No redirect handler; never forward credentials to a second destination.
        return response.status, response.read(65537)
    finally:
        connection.close()


def response_result(status, body, package_id):
    if 200 <= status < 300:
        try:
            ack = json.loads(body) if len(body) <= 65536 else None
        except (ValueError, UnicodeError):
            ack = None
        if isinstance(ack, dict) and ack.get("stored") is True and ack.get("package_id") == package_id and ack.get("sha256") == package_id:
            return "uploaded", "receiver_confirmed", {"stored": True, "package_id": package_id, "sha256": package_id}
        return "rejected", "invalid_receiver_ack", None
    if status in (401, 403):
        return "auth_blocked", "receiver_auth_rejected", None
    if status in (408, 425, 429) or 500 <= status <= 599:
        return "retry_wait", "retryable_http_" + str(status), None
    return "rejected", "http_" + str(status), None


def read_job(path):
    job = read_json(path)
    require(isinstance(job, dict) and type(job.get("schema_version")) is int
            and job["schema_version"] == 1, "invalid_job_schema")
    pid = job.get("package_id")
    require(isinstance(pid, str) and re.fullmatch(r"[0-9a-f]{64}", pid) and Path(path).stem == pid, "invalid_job_id")
    require(isinstance(job.get("state"), str) and job["state"] in STATES, "invalid_job_state")
    for key in ("record_id", "record_revision"):
        require(nonblank(job.get(key)), "invalid_job_" + key)
    for key in ("attempts", "total_attempts"):
        require(type(job.get(key)) is int and job[key] >= 0, "invalid_job_" + key)
    require(job["total_attempts"] >= job["attempts"], "invalid_job_attempt_totals")
    value = job.get("next_attempt_at")
    require((type(value) is int and value >= 0)
            or (type(value) is float and math.isfinite(value) and value >= 0), "invalid_job_next_attempt_at")
    require(isinstance(job.get("history"), list) and all(isinstance(e, dict) for e in job["history"]), "invalid_job_history")
    require(job.get("endpoint") is None or nonblank(job["endpoint"]), "invalid_job_endpoint")
    return job


def send(config_path, outbox, limit=5):
    require(type(limit) is int and 1 <= limit <= 100, "limit_must_be_1_to_100")
    upload_config(config_path)  # Disabled or unconfigured: no queue changes and no network.
    results, handled = [], 0
    with queue_lock(outbox) as box:
        for job_path in sorted((box / "jobs").glob("*.json")):
            if handled >= limit:
                break
            try:
                job = read_job(job_path)
            except (UploadError, OSError, ValueError, TypeError):
                results.append({"job_file": job_path.name, "state": "invalid", "reason": "invalid_job_record"})
                continue  # Preserve the damaged record, but do not starve independent valid jobs.
            if job.get("state") not in {"queued", "retry_wait"} or job.get("next_attempt_at", 0) > time.time():
                continue
            config, token = upload_config(config_path)  # Observe opt-out before each new request.
            handled += 1
            package_id = job.get("package_id", "")
            require(re.fullmatch(r"[0-9a-f]{64}", package_id) and job_path.stem == package_id, "invalid_job_id")
            endpoint = config["endpoint"]
            require(job.get("endpoint") in (None, endpoint), "job_destination_changed: retain original destination")
            require(type(job.get("attempts")) is int and job["attempts"] >= 0, "invalid_job_attempts")
            now, receipt = time.time(), None
            if job["attempts"] >= config["max_attempts"]:
                state, reason = "exhausted", "attempt_limit_reached"
            else:
                package = box / "packages" / (package_id + ".zip")
                try:
                    stream = package.open("rb")
                except OSError:
                    stream = None
                if stream is None:
                    state, reason = "corrupt", "package_missing_or_unreadable"
                else:
                    with stream:
                        if digest_stream(stream) != package_id:
                            state, reason = "corrupt", "package_hash_mismatch"
                        else:
                            try:
                                validate_package(stream, job)
                            except UploadError as exc:
                                state, reason = "needs_review", str(exc)
                            except (OSError, ValueError, KeyError, TypeError, RuntimeError, NotImplementedError, zipfile.BadZipFile):
                                state, reason = "needs_review", "package_contract_requires_review"
                            else:
                                # Packaging/verification can take time; observe a revocation again before POST.
                                config, token = upload_config(config_path)
                                require(config["endpoint"] == endpoint, "destination_changed_during_validation")
                                if job["attempts"] >= config["max_attempts"]:
                                    state, reason = "exhausted", "attempt_limit_reached"
                                else:
                                    job["attempts"] += 1
                                    job["total_attempts"] += 1
                                    job["endpoint"] = endpoint
                                    # Save BEFORE the POST: after a crash, retry with the same idempotency key.
                                    job["state"] = "retry_wait"
                                    job["next_attempt_at"] = now + config["retry_delay_seconds"] * min(2 ** (job["attempts"] - 1), 32)
                                    job["history"].append({"at": now, "event": "request_started", "attempt": job["total_attempts"]})
                                    atomic_json(job_path, job)
                                    try:
                                        status, body = post_bundle(endpoint, token, stream, os.fstat(stream.fileno()).st_size,
                                                                   package_id, config["timeout_seconds"])
                                        state, reason, receipt = response_result(status, body, package_id)
                                    except (OSError, http.client.HTTPException):
                                        state, reason = "retry_wait", "transport_error"
                                    if state == "retry_wait" and job["attempts"] >= config["max_attempts"]:
                                        state = "exhausted"
            job.update(state=state, last_result=reason, updated_at=time.time())
            if receipt:
                job["receipt"] = receipt
                job["uploaded_at"] = time.time()
            job["history"].append({"at": time.time(), "event": state, "reason": reason})
            atomic_json(job_path, job)
            results.append({"package_id": package_id, "state": state, "reason": reason})
            if state in {"auth_blocked", "rejected"}:
                break  # Fix auth/protocol before wasting requests on the rest of this batch.
    return results


def retry(outbox, package_id):
    require(isinstance(package_id, str) and re.fullmatch(r"[0-9a-f]{64}", package_id), "invalid_package_id")
    with queue_lock(outbox) as box:
        path = box / "jobs" / (package_id + ".json")
        job = read_job(path)
        require(job.get("state") in {"exhausted", "auth_blocked", "rejected"}, "job_not_manually_retryable")
        job.update(state="queued", attempts=0, next_attempt_at=0)
        job["history"].append({"at": time.time(), "event": "manual_retry"})
        atomic_json(path, job)
        return {"package_id": package_id, "state": "queued"}


def status(outbox):
    jobs = []
    for path in sorted((Path(outbox) / "jobs").glob("*.json")):
        try:
            jobs.append(read_job(path))
        except (UploadError, OSError, ValueError, TypeError):
            jobs.append({"job_file": path.name, "state": "invalid", "last_result": "invalid_job_record"})
    return {"total": len(jobs), "counts": {state: sum(j.get("state") == state for j in jobs)
                                           for state in sorted(STATES | {"invalid"})},
            "jobs": [{key: j.get(key) for key in ("job_file", "package_id", "record_id", "record_revision", "state",
                                                  "attempts", "total_attempts", "next_attempt_at", "last_result")}
                     for j in jobs]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pack", "send", "status", "retry"):
        command = commands.add_parser(name)
        command.add_argument("--outbox", required=True)
        if name == "pack":
            command.add_argument("--export-root", required=True)
            command.add_argument("--manifest", required=True)
        if name == "send":
            command.add_argument("--config", required=True)
            command.add_argument("--limit", type=int, default=5)
        if name == "retry":
            command.add_argument("--package-id", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if "manifest" in args:
        args["manifest_path"] = args.pop("manifest")
    if "config" in args:
        args["config_path"] = args.pop("config")
    try:
        result = {"pack": pack, "send": send, "status": status, "retry": retry}[command](**args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (UploadError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # Do not print network responses, credentials, or arbitrary file contents.
        message = str(exc) if isinstance(exc, UploadError) else type(exc).__name__
        print(json.dumps({"error": message}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
