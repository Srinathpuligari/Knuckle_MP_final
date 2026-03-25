"""HTTP API for the simplified knuckle verification backend."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import cgi
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from backend.config import (
    BACKEND_ADMIN_CODE,
    BACKEND_HOST,
    BACKEND_PORT,
    IDENTIFY_MARGIN_THRESHOLD,
    IDENTIFY_THRESHOLD,
    VERIFY_MARGIN_THRESHOLD,
)
from backend.engine import KnuckleVerificationEngine
from backend.storage import RegistrationStore


class KnuckleAPIHandler(BaseHTTPRequestHandler):
    """REST API compatible with the existing frontend."""

    engine: KnuckleVerificationEngine
    store: RegistrationStore
    admin_code: str

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status_code: int, message: str) -> None:
        self._send_json(status_code, {"status": "error", "message": message})

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON request body.") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object.")
        return data

    def _read_multipart(self) -> cgi.FieldStorage:
        content_type = self.headers.get("Content-Type", "")
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ,
            keep_blank_values=True,
        )

    def _form_value(self, form: cgi.FieldStorage, name: str) -> str:
        value = form.getfirst(name, "")
        return value.strip() if isinstance(value, str) else ""

    def _form_files(self, form: cgi.FieldStorage, name: str) -> list[bytes]:
        if name not in form:
            return []
        field = form[name]
        items = field if isinstance(field, list) else [field]
        image_bytes: list[bytes] = []
        for item in items:
            if not getattr(item, "file", None):
                continue
            data = item.file.read()
            if data:
                image_bytes.append(data)
        return image_bytes

    def _require_admin(self, payload: dict[str, object]) -> bool:
        supplied_code = str(payload.get("code", "")).strip()
        return bool(supplied_code and supplied_code == self.admin_code)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "message": "Knuckle backend is running.",
                    "registered_users": self.store.count_users(),
                    "feature_version": self.engine.feature_version,
                },
            )
            return
        if parsed.path == "/users":
            self._send_json(
                HTTPStatus.OK,
                {"users": [{"uid": user["uid"], "name": user["name"]} for user in self.store.list_users()]},
            )
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/register":
                self._handle_register()
                return
            if parsed.path == "/verify":
                self._handle_verify()
                return
            if parsed.path == "/identify":
                self._handle_identify()
                return
            if parsed.path == "/admin/users":
                self._handle_admin_users()
                return
            if parsed.path.startswith("/admin/delete/"):
                self._handle_admin_delete(parsed.path.rsplit("/", 1)[-1])
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Server error: {exc}")

    def _handle_register(self) -> None:
        form = self._read_multipart()
        metadata = {
            "name": self._form_value(form, "name"),
            "phone": self._form_value(form, "phone"),
            "email": self._form_value(form, "email"),
            "dob": self._form_value(form, "dob"),
            "address": self._form_value(form, "address"),
            "gender": self._form_value(form, "gender"),
        }
        image_payloads = self._form_files(form, "images")

        if len(metadata["name"]) < 2:
            raise ValueError("Enter a valid full name.")
        if len(metadata["phone"]) < 10:
            raise ValueError("Enter a valid phone number.")

        processed_images = self.engine.process_uploaded_images(image_payloads)
        enrollment = self.engine.create_enrollment_bundle(processed_images)
        uid = self.store.generate_uid()
        self.store.save_registration(uid, metadata, image_payloads, enrollment)

        self._send_json(
            HTTPStatus.CREATED,
            {
                "status": "success",
                "uid": uid,
                "message": "Registration completed. This UID is now linked only to the registered knuckle.",
                "quality": enrollment.registration_quality,
                "images_used": len(processed_images),
            },
        )

    def _handle_verify(self) -> None:
        form = self._read_multipart()
        uid = self._form_value(form, "uid")
        image_payloads = self._form_files(form, "images")

        if not uid:
            raise ValueError("UID is required for verification.")

        claimed_user = self.store.get_user(uid)
        if claimed_user is None:
            raise FileNotFoundError("UID not found in the registered database.")

        query_captures = self.engine.process_uploaded_images(image_payloads)
        candidate_results = []
        for candidate_user, candidate_enrollment in self.store.iter_enrollments():
            result = self.engine.compare_with_enrollment(query_captures, candidate_enrollment)
            candidate_results.append((candidate_user, result))

        if not candidate_results:
            raise FileNotFoundError("No enrolled users are available for verification.")

        candidate_results.sort(key=lambda item: float(item[1]["score"]), reverse=True)
        claimed_entry = next((item for item in candidate_results if item[0].uid == uid), None)
        if claimed_entry is None:
            raise FileNotFoundError("Stored enrollment data is missing for this UID.")

        claimed_user, claimed_result = claimed_entry
        top_user, top_result = candidate_results[0]
        strongest_other_score = max(
            (
                float(candidate_result["score"])
                for candidate_user, candidate_result in candidate_results
                if candidate_user.uid != uid
            ),
            default=0.0,
        )
        claim_margin = float(claimed_result["score"]) - strongest_other_score
        claimed_is_top = top_user.uid == uid
        accepted = bool(
            claimed_result["match"]
            and claimed_is_top
            and float(claimed_result["score"]) >= float(claimed_result["fused_threshold"])
            and claim_margin >= VERIFY_MARGIN_THRESHOLD
        )

        if accepted:
            message = "Knuckle matched successfully for this UID."
        elif not claimed_result["match"]:
            message = "Knuckle not matched for this UID. Use the same registered knuckle and try again."
        elif not claimed_is_top:
            message = "Verification rejected because this capture is closer to another enrolled template."
        else:
            message = "Verification rejected because the claimed match is not separated enough from other users."

        self._send_json(
            HTTPStatus.OK,
            {
                "status": "success",
                "uid": claimed_user.uid,
                "name": claimed_user.name,
                "match": accepted,
                "score": claimed_result["score"],
                "cosine_score": claimed_result["cosine_score"],
                "orb_score": claimed_result["orb_score"],
                "mean_image_score": claimed_result["mean_image_score"],
                "max_score": claimed_result["max_image_score"],
                "best_score": top_result["score"],
                "pass_ratio": claimed_result["pass_ratio"],
                "claim_margin": claim_margin,
                "claimed_is_top_match": claimed_is_top,
                "message": message,
            },
        )

    def _handle_identify(self) -> None:
        query_captures = self.engine.process_uploaded_images(self._form_files(self._read_multipart(), "images"))
        candidates = self.store.iter_enrollments()
        if not candidates:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "success",
                    "found": False,
                    "message": "No registered users are available in the database yet.",
                },
            )
            return

        candidate_results = []
        for user, enrollment in candidates:
            result = self.engine.compare_with_enrollment(query_captures, enrollment)
            candidate_results.append((user, result))

        candidate_results.sort(key=lambda item: float(item[1]["score"]), reverse=True)
        best_user, best_result = candidate_results[0]
        second_best_score = float(candidate_results[1][1]["score"]) if len(candidate_results) > 1 else 0.0
        decision_margin = float(best_result["score"]) - second_best_score
        found = bool(
            best_result["match"]
            and float(best_result["score"]) >= IDENTIFY_THRESHOLD
            and decision_margin >= IDENTIFY_MARGIN_THRESHOLD
        )

        if found:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "success",
                    "found": True,
                    "uid": best_user.uid,
                    "name": best_user.name,
                    "score": best_result["score"],
                    "best_score": best_result["max_image_score"],
                    "decision_margin": decision_margin,
                    "message": "Matching knuckle found in the registered database.",
                },
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "status": "success",
                "found": False,
                "score": best_result["score"],
                "best_score": best_result["max_image_score"],
                "decision_margin": decision_margin,
                "message": "Knuckle not found in the registered database.",
            },
        )

    def _handle_admin_users(self) -> None:
        payload = self._read_json()
        if not self._require_admin(payload):
            self._send_error(HTTPStatus.FORBIDDEN, "Invalid admin access code.")
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "status": "success",
                "total": self.store.count_users(),
                "users": self.store.list_users(),
            },
        )

    def _handle_admin_delete(self, uid: str) -> None:
        payload = self._read_json()
        if not self._require_admin(payload):
            self._send_error(HTTPStatus.FORBIDDEN, "Invalid admin access code.")
            return

        deleted = self.store.delete_user(uid)
        if not deleted:
            self._send_error(HTTPStatus.NOT_FOUND, "User not found.")
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "status": "success",
                "message": f"Deleted user {uid} successfully.",
            },
        )


def build_handler(
    engine: KnuckleVerificationEngine,
    store: RegistrationStore,
    admin_code: str,
) -> type[KnuckleAPIHandler]:
    class ConfiguredHandler(KnuckleAPIHandler):
        pass

    ConfiguredHandler.engine = engine
    ConfiguredHandler.store = store
    ConfiguredHandler.admin_code = admin_code
    return ConfiguredHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the knuckle biometric backend server.")
    parser.add_argument("--host", default=BACKEND_HOST, help="Host to bind the API server to.")
    parser.add_argument("--port", type=int, default=BACKEND_PORT, help="Port to bind the API server to.")
    parser.add_argument("--admin-code", default=BACKEND_ADMIN_CODE, help="Admin access code for the frontend.")
    args = parser.parse_args()

    engine = KnuckleVerificationEngine()
    store = RegistrationStore()
    handler_class = build_handler(engine, store, args.admin_code)
    server = ThreadingHTTPServer((args.host, args.port), handler_class)

    print(f"Knuckle backend running on http://{args.host}:{args.port}")
    print(f"Registered users: {store.count_users()}")
    print(f"Feature version: {engine.feature_version}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down knuckle backend.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
