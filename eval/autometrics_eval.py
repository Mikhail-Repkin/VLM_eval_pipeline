from bleurt import score
from tqdm import tqdm

import pandas as pd


class TextEvaluator:
    """Оценка качества сгенерированного текста через автометрики."""
    def __init__(self, bleurt_checkpoint: str = "./BLEURT-20"):
        """
        https://github.com/google-research/bleurt.git
        https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip

        :param bleurt_checkpoint: Путь к модели BLEURT ('./bleurt-20').
        """
        self.bleurt_checkpoint = bleurt_checkpoint
        # self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1',
        #                                               'rouge2',
        #                                               'rougeL'
        #                                               ], use_stemmer=False)
        self.bleurt_scorer = score.BleurtScorer(self.bleurt_checkpoint)

    def eval_autometrics(self, candidate: str, reference: str) -> dict:
        """
        Оценивает качество сгенерированного текста (candidate)
        относительно эталонного (reference).

        :param candidate: Сгенерированный текст.
        :param reference: Эталонный (референсный) текст.

        :return: dict с оценками.
        """
        # # Вычисление метрик ROUGE
        # rouge_scores = self.rouge_scorer.score(reference, candidate)

        # Вычисление метрики BLEURT
        bleurt_result = self.bleurt_scorer.score(
            references=[reference], candidates=[candidate]
        )[0]

        return {
            # "rouge1": rouge_scores['rouge1'].fmeasure,
            # "rouge2": rouge_scores['rouge2'].fmeasure,
            # "rougeL": rouge_scores['rougeL'].fmeasure,
            "bleurt": bleurt_result
        }

    def eval_autometrics_texts(
        self, df: pd.DataFrame, output_path: str = "auto_metrics_stats.xlsx"
    ):
        """
        Оценивает набор текстов по метрикам и сохраняет результаты в файл.

        :param df: DataFrame с текстами для оценки.
                   Ожидается ['Image', 'model_name', 'candidate', 'reference'].
        :param output_path: Путь для сохранения итогового Excel-файла.
        """
        required_columns = {"Image", "model_name", "candidate", "reference"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame должен содержать колонки: "
                             f"{required_columns}")

        results = []

        for _, row in tqdm(df.iterrows(), desc="Оценка метрик", total=len(df)):
            candidate = row["candidate"]
            reference = row["reference"]
            img = row["Image"]

            metrics = self.eval_autometrics(candidate, reference)

            res = {
                "Image": img,
                "model_name": row["model_name"],
                "candidate": candidate,
                "reference": reference,
            }

            res.update(metrics)
            results.append(res)

        # Создаем итоговый DataFrame
        final_stats = pd.DataFrame(results)
        final_stats.to_excel(output_path, index=False)

        print(f"Результаты успешно сохранены в файл: {output_path}")
