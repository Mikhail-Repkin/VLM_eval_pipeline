import regex
from tqdm import tqdm
import pandas as pd
from pyaspeller import YandexSpeller

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SpellChecker:
    """Класс для проверки орфографии текста с помощью Yandex Speller."""
    def __init__(self, lang="ru"):
        self.speller = YandexSpeller(lang=lang)

    def check_spelling(self, text):
        """
        Возвращает количество ошибок и информацию об ошибках.
        """
        errors = list(self.speller.spell(text))
        error_count = len(errors)
        error_words = [error["word"] for error in errors]
        suggestions = [error["s"] for error in errors]
        return error_count, error_words, suggestions

    def analyze_texts(self, texts):
        """
        Анализирует список текстов на орфографические ошибки.
        Возвращает DataFrame с результатами.
        """
        results = []

        for idx, text in enumerate(
            tqdm(texts, desc="Проверка орфографии", unit="текст")
        ):
            error_count, error_words, suggestions = self.check_spelling(text)
            results.append({"Errors (cnt)": error_count,
                            "Error Words": error_words})

        results_df = pd.DataFrame(results)
        return results_df

    def analyze_file(self, file_path):
        """
        Анализирует текстовый файл на орфографические ошибки.
        Каждая строка файла считается отдельным текстом.
        Возвращает DataFrame с результатами.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        return self.analyze_texts(lines)


def count_non_cyrillic_and_non_latin_letters(text):
    """
    Подсчитывает количество не ru/en букв в тексте,
    исключая цифры и знаки препинания.
    """
    # Находим все буквы в тексте
    letters = regex.findall(r"\p{L}", text)
    # Фильтруем только те, которые не являются кириллическими или латинскими
    non_cyrillic_non_latin_letters = [
        letter
        for letter in letters
        if not (
            regex.match(r"\p{IsCyrillic}", letter)
            or regex.match(r"\p{IsLatin}", letter)
        )
    ]

    return len(non_cyrillic_non_latin_letters)


def count_foreign_letters_in_texts(texts):
    """
    Принимает список текстов и возвращает DataFrame с кол-вом не ru/en букв.
    """
    non_cyrillic_non_latin_counts = []

    for text in tqdm(texts, desc="Подсчет символов", unit="текст"):
        non_cyrillic_non_latin_counts.append(
            count_non_cyrillic_and_non_latin_letters(text)
        )

    df = pd.DataFrame(
        {"Non cyrillic and non latin chars (cnt)":
         non_cyrillic_non_latin_counts}
    )
    return df


def calculate_perplexity_for_texts(texts):
    """
    Вычисляет перплексию для списка текстов.
    Возвращает датафрейм с текстами и перплексиями.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "ai-forever/rugpt3large_based_on_gpt2"
        )
    model = AutoModelForCausalLM.from_pretrained(
        "ai-forever/rugpt3large_based_on_gpt2"
        )
    model.eval()

    data = []
    for text in tqdm(texts, desc="Вычисление перплексии", unit="текст"):
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)

        data.append({"Perplexity": perplexity.item()})

    return pd.DataFrame(data)
