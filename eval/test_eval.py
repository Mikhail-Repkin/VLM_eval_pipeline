import os
import time
import subprocess
import requests
import base64
from tqdm import tqdm
from openai import OpenAI

import pandas as pd


class ImageDescriptionGenerator:
    """Запуск модели через vllm и генерация ответа."""
    def __init__(
        self,
        model_name,
        system_prompt,
        user_prompt,
        port=8000,
        log_file="vllm.log"
    ):
        self.model_name = model_name
        self.port = port
        self.log_file = log_file
        self.api_key = "EMPTY"
        self.api_base = f"http://localhost:{port}/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.server_process = None
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

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
                #  "--max_model_len", "15000"  # только для Phi-3
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

    @staticmethod
    def encode_base64_content_from_file(file_path):
        """Кодирует содержимое файла в base64."""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def generate_description(self, image_url, temperature):
        """Генерирует описание изображения для заданной температуры."""
        chat_completion = self.client.chat.completions.create(
            messages=[
                # {"role": "system",
                #  "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {"type": "text", "text": self.user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                },
            ],
            model=self.model_name,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content

    def generate_descriptions_for_images(
        self, image_paths, temperatures, base64=False, num_variants=5
    ):
        """
        Генерирует описания для нескольких изображений с разными температурами.
        """
        print("Генерируем описания изображений")
        results = []

        for img_name, image_path in image_paths.items():
            if base64 is True:
                image_base64 = self.encode_base64_content_from_file(image_path)
                img = f"data:image/jpeg;base64,{image_base64}"
            else:
                img = image_path

            for temperature in temperatures:
                generation_times = []
                text_lengths = []
                char_speeds = []

                print(
                    f"\nГенерация {num_variants} вариантов для изображения: "
                    f"{image_path}, температура: {temperature}"
                )
                for _ in tqdm(
                    range(num_variants), desc="Обработка варианта", leave=True
                ):
                    start_time = time.time()
                    description = self.generate_description(img, temperature)
                    end_time = time.time()
                    print(description)

                    # Расчет времени генерации
                    time_taken = round(end_time - start_time, 1)

                    # Длина текста в символах
                    text_length = len(description)

                    # Скорость генерации в символах в секунду
                    char_speed = round(text_length / time_taken, 1)

                    generation_times.append(time_taken)
                    text_lengths.append(text_length)
                    char_speeds.append(char_speed)

                    results.append(
                        {
                            "Image": img_name,
                            "Temperature": temperature,
                            "Generated Text": description,
                            "Time Taken (sec)": time_taken,
                            "Text Length (char)": text_length,
                            "Chars/sec": char_speed,
                        }
                    )

        # Преобразовать результаты в DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def stop_session(self):
        """Перезапускаем сеанс."""
        print("Перезагрузка сеанса для полной очистки ресурсов...")
        os.system("kill -9 -1")
