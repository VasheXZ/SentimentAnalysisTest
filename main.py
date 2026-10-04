from abc import ABC, abstractmethod

from transformers import pipeline


# =========================================================
# МОДЕЛИ
# =========================================================

class EmotionModel(ABC):
    """Общий интерфейс модели определения эмоций."""

    @abstractmethod
    def predict(self, text: str):
        pass


class HuggingFaceEmotionModel(EmotionModel):
    """Базовая реализация модели через Hugging Face."""

    MODEL_NAME = ""

    def __init__(self):
        print(f"Загрузка модели: {self.MODEL_NAME}")
        self.classifier = pipeline(
            "text-classification",
            model=self.MODEL_NAME
        )

    def predict(self, text: str):
        results = self.classifier(
            text,
            top_k=None,
            truncation=True
        )

        # Для совместимости с разными версиями transformers
        if results and isinstance(results[0], list):
            results = results[0]

        return sorted(
            results,
            key=lambda item: item["score"],
            reverse=True
        )


class TinyRuBertEmotionModel(HuggingFaceEmotionModel):
    MODEL_NAME = (
        "seara/"
        "rubert-tiny2-russian-emotion-detection-ru-go-emotions"
    )


class BaseRuBertEmotionModel(HuggingFaceEmotionModel):
    MODEL_NAME = (
        "seara/"
        "rubert-base-cased-russian-emotion-detection-ru-go-emotions"
    )


# =========================================================
# FACTORY METHOD — создание модели
# =========================================================

class EmotionModelFactory(ABC):

    @abstractmethod
    def create_model(self) -> EmotionModel:
        pass


class TinyModelFactory(EmotionModelFactory):

    def create_model(self) -> EmotionModel:
        return TinyRuBertEmotionModel()


class BaseModelFactory(EmotionModelFactory):

    def create_model(self) -> EmotionModel:
        return BaseRuBertEmotionModel()


# =========================================================
# STRATEGY — преобразование текста
# =========================================================

class TextTransformationStrategy(ABC):

    @abstractmethod
    def transform(self, text: str) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class TypoStrategy(TextTransformationStrategy):
    """Создаёт одну искусственную опечатку."""

    def transform(self, text: str) -> str:
        words = text.split()

        for i, word in enumerate(words):
            clean_word = word.strip(".,!?;:")

            if len(clean_word) >= 6:
                position = len(clean_word) // 2 - 1

                changed = (
                    clean_word[:position]
                    + clean_word[position + 1]
                    + clean_word[position]
                    + clean_word[position + 2:]
                )

                words[i] = word.replace(clean_word, changed, 1)
                break

        return " ".join(words)

    def name(self) -> str:
        return "Опечатка"


class AbbreviationStrategy(TextTransformationStrategy):
    """Заменяет некоторые слова разговорными сокращениями."""

    def transform(self, text: str) -> str:
        replacements = {
            "очень": "оч",
            "сейчас": "щас",
            "пожалуйста": "пж"
        }

        result = text

        for original, short in replacements.items():
            result = result.replace(original, short)

        return result

    def name(self) -> str:
        return "Сокращения"


class EmojiStrategy(TextTransformationStrategy):
    """Добавляет к сообщению эмоциональный сигнал."""

    def __init__(self, emoji="😢"):
        self.emoji = emoji

    def transform(self, text: str) -> str:
        return f"{text} {self.emoji}"

    def name(self) -> str:
        return "Эмодзи"


# =========================================================
# FACADE — проведение эксперимента
# =========================================================

class ExperimentFacade:
    """Предоставляет простой интерфейс для проведения эксперимента."""

    def __init__(
        self,
        model: EmotionModel,
        strategy: TextTransformationStrategy
    ):
        self.model = model
        self.strategy = strategy

    def set_strategy(self, strategy: TextTransformationStrategy):
        self.strategy = strategy

    @staticmethod
    def print_emotions(results, count=6):
        for item in results[:count]:
            print(
                f"  {item['label']:<15}"
                f"{item['score']:.3f}"
            )

    def run(self, text: str):
        changed_text = self.strategy.transform(text)

        original_result = self.model.predict(text)
        changed_result = self.model.predict(changed_text)

        print("\n" + "=" * 60)
        print(f"Преобразование: {self.strategy.name()}")

        print("\nИсходный текст:")
        print(text)

        print("\nОсновные эмоции:")
        self.print_emotions(original_result)

        print("\nИзменённый текст:")
        print(changed_text)

        print("\nОсновные эмоции после изменения:")
        self.print_emotions(changed_result)


# =========================================================
# ТЕСТОВАЯ ПРОГРАММА
# =========================================================

if __name__ == "__main__":

    # Для быстрой проверки используем Tiny.
    # Чтобы проверить большую модель, заменить на BaseModelFactory().
    factory = BaseModelFactory()

    model = factory.create_model()

    strategies = [
        TypoStrategy(),
        AbbreviationStrategy(),
        EmojiStrategy("😢")
    ]

    texts = [
    "Я очень рад, что всё наконец получилось!",
    "Ну вот и всё.",
    "Я получил результаты экзамена.",
    "Не ожидал, что всё закончится именно так.",
]

    experiment = ExperimentFacade(
        model=model,
        strategy=strategies[0]
    )

    for text in texts:
        print("\n\n" + "#" * 60)
        print(f"ТЕСТОВЫЙ ТЕКСТ: {text}")
        print("#" * 60)

        for strategy in strategies:
            experiment.set_strategy(strategy)
            experiment.run(text)