#!/usr/bin/env python3
"""
Web server cho công cụ lồng tiếng AI (ai.py)
- Yêu cầu mật khẩu để đăng nhập (WEB_PASSWORD)
- Upload video → chạy pipeline lồng tiếng dưới nền (không block request)
- Theo dõi tiến trình qua /status
- Tải video kết quả qua /download
"""
import http.server
import json
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path

WORK_DIR = os.environ.get("WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
WEB_PASSWORD = os.environ.get("WEB_PASSWORD", "")
SESSION_TTL = 6 * 3600

UPLOAD_DIR = Path(WORK_DIR) / "uploads"
OUTPUT_DIR = Path(WORK_DIR) / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_sessions = {}
_failed_logins = {}
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 60

# Trạng thái các job lồng tiếng đang chạy: job_id -> {status, progress, log, output_file, error}
_jobs = {}
_jobs_lock = threading.Lock()


def new_session():
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + SESSION_TTL
    return token


def is_valid_session(token):
    exp = _sessions.get(token)
    if exp is None:
        return False
    if time.time() > exp:
        _sessions.pop(token, None)
        return False
    return True


def safe_filename(name: str) -> str:
    """Chỉ giữ ký tự an toàn trong tên file, chống path traversal."""
    name = os.path.basename(name)
    name = re.sub(r'[^A-Za-z0-9._\-]', '_', name)
    return name or f"file_{int(time.time())}"


def run_dubbing_job(job_id: str, input_path: str, voice: str, model_size: str):
    """Chạy ai.py trong tiến trình con, cập nhật trạng thái job khi có log mới."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    output_path = str(OUTPUT_DIR / f"{job_id}_vi.mp4")
    cmd = [
        "python3", os.path.join(WORK_DIR, "ai.py"),
        input_path,
        "--voice", voice,
        "--model", model_size,
        "--output", output_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=WORK_DIR,
        )
        log_lines = []
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            with _jobs_lock:
                _jobs[job_id]["log"] = "\n".join(log_lines[-200:])
        proc.wait()

        with _jobs_lock:
            if proc.returncode == 0 and os.path.exists(output_path):
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["output_file"] = f"{job_id}_vi.mp4"
            else:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = f"Pipeline thoát với mã lỗi {proc.returncode}"
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


class APIHandler(http.server.SimpleHTTPRequestHandler):

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        return is_valid_session(auth[len("Bearer "):])

    def _require_auth(self):
        if not self._authorized():
            self._send_json(401, {"error": "Chưa đăng nhập hoặc token hết hạn"})
            return False
        return True

    def _client_ip(self):
        return self.client_address[0]

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/":
            self.path = "/login.html"
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

        if parsed.path == "/status":
            if not self._require_auth():
                return
            params = urllib.parse.parse_qs(parsed.query)
            job_id = params.get("job_id", [""])[0]
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None:
                self._send_json(404, {"error": "Không tìm thấy job"})
                return
            self._send_json(200, job)
            return

        if parsed.path == "/jobs":
            if not self._require_auth():
                return
            with _jobs_lock:
                summary = [
                    {"job_id": jid, "status": j["status"], "input_name": j.get("input_name", "")}
                    for jid, j in _jobs.items()
                ]
            self._send_json(200, sorted(summary, key=lambda x: x["job_id"], reverse=True))
            return

        if parsed.path == "/download":
            if not self._require_auth():
                return
            params = urllib.parse.parse_qs(parsed.query)
            job_id = params.get("job_id", [""])[0]
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None or job.get("status") != "done":
                self._send_json(404, {"error": "Video chưa sẵn sàng hoặc job không tồn tại"})
                return
            filepath = OUTPUT_DIR / job["output_file"]
            if not filepath.exists():
                self._send_json(404, {"error": "Không tìm thấy file video"})
                return
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Disposition", f"attachment; filename={job['output_file']}")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/login":
            ip = self._client_ip()
            now = time.time()
            attempts = [t for t in _failed_logins.get(ip, []) if now - t < LOGIN_LOCKOUT_SECONDS]
            if len(attempts) >= MAX_LOGIN_ATTEMPTS:
                self._send_json(429, {"error": f"Quá nhiều lần sai, thử lại sau {LOGIN_LOCKOUT_SECONDS}s"})
                return

            content_length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(content_length) or b"{}")
            except json.JSONDecodeError:
                data = {}

            if not WEB_PASSWORD:
                self._send_json(500, {"error": "Server chưa cấu hình WEB_PASSWORD"})
                return
            if secrets.compare_digest(data.get("password", ""), WEB_PASSWORD):
                _failed_logins.pop(ip, None)
                token = new_session()
                self._send_json(200, {"token": token, "expires_in": SESSION_TTL})
            else:
                _failed_logins.setdefault(ip, []).append(now)
                self._send_json(401, {"error": "Sai mật khẩu"})
            return

        if parsed.path == "/logout":
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                _sessions.pop(auth[len("Bearer "):], None)
            self._send_json(200, {"message": "Đã đăng xuất"})
            return

        if not self._require_auth():
            return

        if parsed.path == "/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 500 * 1024 * 1024:  # giới hạn 500MB
                self._send_json(413, {"error": "File quá lớn (giới hạn 500MB)"})
                return

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json(400, {"error": "Cần gửi dạng multipart/form-data"})
                return

            boundary = content_type.split("boundary=")[-1].encode()
            body = self.rfile.read(content_length)

            filename, file_data = None, None
            for part in body.split(b"--" + boundary):
                if b"Content-Disposition" not in part:
                    continue
                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue
                headers_part = part[:header_end].decode(errors="ignore")
                m = re.search(r'filename="([^"]+)"', headers_part)
                if not m:
                    continue
                filename = m.group(1)
                file_data = part[header_end + 4:]
                if file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]
                break

            if not filename or file_data is None:
                self._send_json(400, {"error": "Không tìm thấy file trong request"})
                return

            safe_name = safe_filename(filename)
            job_id = f"{int(time.time())}_{secrets.token_hex(4)}"
            input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
            with open(input_path, "wb") as f:
                f.write(file_data)

            voice = self.headers.get("X-Voice", "female")
            model_size = self.headers.get("X-Model", "small")
            if voice not in ("female", "male"):
                voice = "female"
            if model_size not in ("tiny", "base", "small", "medium", "large"):
                model_size = "small"

            with _jobs_lock:
                _jobs[job_id] = {
                    "status": "queued", "progress": 0, "log": "",
                    "input_name": safe_name, "output_file": None, "error": None,
                }

            thread = threading.Thread(
                target=run_dubbing_job, args=(job_id, str(input_path), voice, model_size), daemon=True,
            )
            thread.start()

            self._send_json(200, {"job_id": job_id, "message": "Đã bắt đầu xử lý"})
            return

        self._send_json(404, {"error": "Không tìm thấy endpoint"})


if __name__ == "__main__":
    if not WEB_PASSWORD:
        print("⚠️  CẢNH BÁO: WEB_PASSWORD chưa được đặt — mọi đăng nhập sẽ bị từ chối.")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    PORT = int(os.environ.get("PORT", 9999))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), APIHandler)
    print(f"✅ AI Dubbing server đang chạy tại port {PORT}")
    server.serve_forever()
