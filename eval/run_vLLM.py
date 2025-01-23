import os
import time
import subprocess
import requests
from openai import OpenAI


class RunVLLM:
    """Запуск сервера vllm."""

    def __init__(self, model_name, port=8000, log_file="vllm.log"):
        self.model_name = model_name
        self.port = port
        self.log_file = log_file
        self.api_key = "EMPTY"
        self.api_base = f"http://localhost:{port}/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.server_process = None

    def start_server(self):
        """Запускает сервер в фоновом режиме."""
        self.server_process = subprocess.Popen(
            [
                "vllm",
                "serve",
                self.model_name,
                "--trust-remote-code",
                "--port",
                str(self.port),
            ],
            stdout=open(self.log_file, "w"),
            stderr=subprocess.STDOUT,
        )
        print(f"Сервер запущен с PID: {self.server_process.pid}")
        self._wait_for_server_ready()

    def _wait_for_server_ready(self, timeout=600, check_interval=60):
        """Ожидает, пока сервер станет доступным, в пределах тайм-аута."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_server():
                print("Сервер готов к работе.")
                return
            print("Ожидание запуска модели...")
            time.sleep(check_interval)
        raise TimeoutError("Сервер не стал доступным.")

    def check_server(self):
        """Проверяет доступность сервера и возвращает."""
        try:
            response = requests.get(f"http://localhost:{self.port}/v1/models")
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        return False

    def stop_session(self):
        """Перезапускаем сеанс."""
        print("Перезагрузка сеанса для полной очистки ресурсов...")
        os.system("kill -9 -1")
