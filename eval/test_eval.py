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
        self, model_name, system_prompt, user_prompt, port=8000, log_file="vllm.log"
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


class VQAGenerator:
    """Генерация ответа по картинке и вопросу."""

    def __init__(self, model_name, port=8000, log_file="vllm.log"):
        self.model_name = model_name
        self.client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

    @staticmethod
    def encode_base64_content_from_file(file_path):
        """Кодирует содержимое файла в base64."""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def generate_description(
        self,
        system_prompt: str,
        user_prompt: str,
        image_url: str,
        temperature: int
    ):
        """Генерирует описание изображения для заданной температуры."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    },
                ],
                model=self.model_name,
                temperature=temperature,
                timeout=10  # Устанавливаем таймаут ответ модели
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            # Любая ошибка приведёт к возврату пустой строки
            print("Отказ генерации", e)
            return ""

    def generate_descriptions_for_images(
        self,
        data: list[dict],
        temperatures: list,
        system_prompt: str,
        base64: bool = False,
        num_variants: int = 5,
    ):
        """
        Генерирует описания для нескольких изображений с разными температурами.

        data:
             {'File_name': '18-01-05.jpg',
              'Question': 'Какие действия предпринять, если я случайно ввел не тот номер телефона?',
              'Answer': 'Вы можете удалить введенный номер, нажав на символ "крестика" в текстовом поле, и ввести правильный номер мобильного телефона.',
              'url': 'https://raw.githubusercontent.com/Mikhail-Repkin/VLM_eval_pipeline/refs/heads/main/open_test_dataset_VQA/phone/18-01-05.jpg',
              'device': 'phone'}
        """
        print("Генерируем описания изображений")

        results = []

        for line in tqdm(data, desc="Обработка вопроса", leave=True):
            # Извлекаем данные объекта
            name = line["File_name"]
            question = line["Question"]
            answer = line["Answer"]
            url = line["url"]
            device = line["device"]

            if base64 is True:
                image_base64 = self.encode_base64_content_from_file(url)
                img = f"data:image/jpeg;base64,{image_base64}"
            else:
                img = url

            for temperature in temperatures:
                text_lengths = []
                char_speeds = []

                print(
                    f"\nГенерация {num_variants} вариантов для изображения: "
                    f"{img}, температура: {temperature}"
                )
                for _ in tqdm(
                    range(num_variants), desc="Обработка варианта", leave=True
                ):
                    start_time = time.time()
                    description = self.generate_description(
                        system_prompt, question, img, temperature
                    )
                    end_time = time.time()
                    print(description)

                    # Расчет времени генерации
                    time_taken = round(end_time - start_time, 1)

                    # Длина текста в символах
                    text_length = len(description)

                    # Скорость генерации в символах в секунду
                    char_speed = round(text_length / time_taken, 1)

                    text_lengths.append(text_length)
                    char_speeds.append(char_speed)

                    results.append(
                        {
                            "Image": name,
                            "Question": question,
                            "Answer": answer,
                            "Device": device,
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
