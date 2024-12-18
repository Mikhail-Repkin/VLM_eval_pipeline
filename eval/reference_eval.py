import requests
import json
from tqdm import tqdm

import pandas as pd


# Примечание: в gpt перезапись системного промта ухудшает качество генерации,
# поэтому DEFAULT_SYSTEM_MESSAGE подавался в модель под ролью пользователя.
class GPT4DescriptionGenerator:
    """Класс для генерации описания/ответов по изображению с помощью GPT-4."""
    def __init__(self, model_name, gpt4_api_key, system_prompt, user_prompt):
        self.model_name = model_name
        self.url = "https://gptunnel.ru/v1/chat/completions"

        if not gpt4_api_key:
            raise ValueError(
                "API key not found. Please set the GPT_TUNNEL_KEY variable."
            )

        self.api_key = gpt4_api_key
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def generate_description(self, image_url, temperature):
        """Генерирует описание для одного изображения."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Формируем запрос к GPTunnel (аналог Chat Completion)
        data = {
            "model": self.model_name,
            "messages": [
                # {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {"type": "text", "text": self.user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": temperature,
            "useWalletBalance": True,
        }

        response = requests.post(
            self.url,
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code == 200:
            resp = response.json()
            print("Total Cost:", resp["usage"]["total_cost"])
            print("Total Tokens:", resp["usage"]["total_tokens"])

            return resp["choices"][0]["message"]["content"]
        else:
            raise Exception(
                f"Request failed with {response.status_code}: {response.text}"
            )

    def generate_descriptions_for_images(
        self, image_urls, temperatures, num_variants=1
    ):
        """
        Генерирует описания для нескольких изображений с разными температурами.
        """
        print("Генерируем описания изображений")
        results = []

        for img_name, image_url in image_urls.items():

            for temperature in temperatures:
                print(
                    f"\nГенерация {num_variants} вариантов для изображения: "
                    f"{image_url}, "
                    f"температура: {temperature}"
                )
                for _ in tqdm(
                    range(num_variants), desc="Обработка варианта", leave=True
                ):
                    description = self.generate_description(image_url,
                                                            temperature)
                    print(description)

                    results.append(
                        {
                            "Image": img_name,
                            "Temperature": temperature,
                            "Generated Text": description,
                        }
                    )

        # Преобразуем результаты в DataFrame
        results_df = pd.DataFrame(results)
        return results_df
