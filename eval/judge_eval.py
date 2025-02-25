import json
import random
import requests


class ModelComparisonJudge:
    """Класс для судейской оценки ответов от двух моделей."""

    def __init__(
        self,
        model,
        gpt4_api_key,
        system_msg,
        user_msg,
        url="https://gptunnel.ru/v1/chat/completions",
    ):
        self.model = model
        self.url = url

        if not gpt4_api_key:
            raise ValueError("API key not found.")

        self.api_key = gpt4_api_key
        self.system_prompt = system_msg
        self.user_msg = user_msg

    def parse_llm_response(self, response_text):
        """Парсит JSON-ответ из текста LLM."""
        try:
            # Извлекаем JSON часть из текста ответа
            json_start_index = response_text.find("{")
            json_end_index = response_text.rfind("}") + 1

            if json_start_index == -1 or json_end_index == -1:
                raise ValueError("JSON data not found in the response text.")

            json_data = response_text[json_start_index:json_end_index]
            parsed_response = json.loads(json_data)
            return parsed_response
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def judge_responses(self, image_url, answer_A, answer_B, max_retries=3):
        """Метод для судейской оценки двух ответов.
        Params:
        - image_url: путь к изображению
        - answer_A: ответ эталонной модели
        - answer_B: ответ тестируемой модели
        - max_retries: макс. количество повторов

        Returns:
        {
            "ModelA_score": int,
            "ModelB_score": int,
            "Explanation": str
        }

        Примечание: Модели не указываются явно судье. Ответы перемешиваются.
        """

        answers = [answer_A, answer_B]
        random.shuffle(answers)  # перемешиваем ответы

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_prompt = (
            "[User Question]\n"
            f"{self.user_msg}\n\n"
            "[The Start of Assistant A's Answer]\n"
            f"{answers[0]}\n"
            "[The End of Assistant A's Answer]\n\n"
            "[The Start of Assistant B's Answer]\n"
            f"{answers[1]}\n"
            "[The End of Assistant B's Answer]"
        )

        # Формируем запрос к GPTunnel (аналог Chat Completion)
        # "detail": "low" will enable the "low res" mode.
        # The model will receive a low-res 512px x 512px version of the image,
        # and represent the image with a budget of 85 tokens.
        # This allows the API to return faster responses and consume
        # fewer input tokens for use cases that do not require high detail.
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "low"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "useWalletBalance": True,
        }

        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=None,
                )
                if response.status_code == 200:
                    resp = response.json()
                    print("Total Cost:", resp["usage"]["total_cost"])
                    print("Total Tokens:", resp["usage"]["total_tokens"])

                    try:
                        result = resp["choices"][0]["message"]["content"]
                        parsed_result = self.parse_llm_response(result)
                        print("result:", result)
                        print("result:", parsed_result)

                        # Идентифицируем расположение ответов
                        if answers[0] == answer_A:
                            # первая модель (А) - эталон, 2я (B) - кандидат
                            final_result = {
                                "Reference_score": parsed_result[
                                    "ModelA_score"
                                    ],
                                "Candidate_score": parsed_result[
                                    "ModelB_score"
                                    ],
                                "Explanation": parsed_result[
                                    "Explanation"
                                    ],
                            }
                        else:
                            # первая модель (А) - кандидат, 2я (B) - эталон
                            final_result = {
                                "Reference_score": parsed_result[
                                    "ModelB_score"
                                    ],
                                "Candidate_score": parsed_result[
                                    "ModelA_score"
                                    ],
                                "Explanation": parsed_result[
                                    "Explanation"
                                    ],
                            }
                        return final_result
                    except ValueError as e:
                        print(f"Parsing error on attempt {retry_count + 1}: "
                              f"{e}")
                        retry_count += 1
            except requests.exceptions.Timeout:
                print("Timeout error occurred, retrying...")
                retry_count += 1
            except Exception as e:
                raise Exception(f"Request failed: {e}")

        raise Exception("Failed to parse LLM response after multiple retries.")


class VQAModelJudge:
    """Класс для судейской оценки ответов на VQA задачу."""

    def __init__(
        self,
        model,
        gpt4_api_key,
        system_msg,
        url="https://gptunnel.ru/v1/chat/completions",
    ):
        self.model = model
        self.url = url

        if not gpt4_api_key:
            raise ValueError("API key not found.")

        self.api_key = gpt4_api_key
        self.system_prompt = system_msg

    def parse_llm_response(self, response_text):
        """Парсит JSON-ответ из текста LLM."""
        try:
            # Извлекаем JSON часть из текста ответа
            json_start_index = response_text.find("{")
            json_end_index = response_text.rfind("}") + 1

            if json_start_index == -1 or json_end_index == -1:
                raise ValueError("JSON data not found in the response text.")

            json_data = response_text[json_start_index:json_end_index]
            parsed_response = json.loads(json_data)
            return parsed_response
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def judge_responses(
        self,
        image_url,
        user_msg,
        answer_reference,
        answer_candidate,
        max_retries=3
    ):
        """Метод для судейской оценки ответа на вопрос об изображении.
        Params:
        - image_url: путь к изображению
        - user_msg: вопрос пользователя
        - answer_reference: эталонный человеческий ответ
        - answer_candidate: ответ тестируемой модели
        - max_retries: макс. количество повторов

        Returns:
        {
            "Candidate_score": int,
            "Explanation": str
        }

        """

        answers = [answer_reference, answer_candidate]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_prompt = (
            "[User question]\n"
            f"{user_msg}\n\n"
            "[The start of Human reference answer]\n"
            f"{answers[0]}\n"
            "[The end of Human reference answer]\n\n"
            "[The start of Candidate model answer]\n"
            f"{answers[1]}\n"
            "[The end of Candidate model answer]"
        )

        # Формируем запрос к GPTunnel (аналог Chat Completion)
        # "detail": "low" will enable the "low res" mode.
        # The model will receive a low-res 512px x 512px version of the image,
        # and represent the image with a budget of 85 tokens.
        # This allows the API to return faster responses and consume
        # fewer input tokens for use cases that do not require high detail.
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "low"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "useWalletBalance": True,
        }

        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=None,
                )
                if response.status_code == 200:
                    resp = response.json()
                    print("Total Cost:", resp["usage"]["total_cost"])
                    print("Total Tokens:", resp["usage"]["total_tokens"])

                    try:
                        result = resp["choices"][0]["message"]["content"]
                        parsed_result = self.parse_llm_response(result)
                        print("result:", result)
                        print("result:", parsed_result)

                        final_result = {
                            "Candidate_score": parsed_result["Candidate_score"],
                            "Explanation": parsed_result["Explanation"],
                        }

                        return final_result
                    except ValueError as e:
                        print(f"Parsing error on attempt {retry_count + 1}: " f"{e}")
                        retry_count += 1
            except requests.exceptions.Timeout:
                print("Timeout error occurred, retrying...")
                retry_count += 1
            except Exception as e:
                raise Exception(f"Request failed: {e}")

        raise Exception("Failed to parse LLM response after multiple retries.")
