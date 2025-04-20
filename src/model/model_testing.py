from typing import Optional, Dict, Any
from logger import Logger

test_logger = Logger(name="model_testing", file_name="model_testing.log").get_logger()


def test_model(model: Any, test_data: Optional[list]) -> Optional[Dict[str, Any]]:
    test_logger.info("Начало тестирования модели.")
    try:
        if model is None:
            test_logger.error("Тестирование невозможно: модель отсутствует.")
            return None

        if test_data is None:
            test_logger.error("Тестирование невозможно: тестовые данные отсутствуют.")
            return None

        if len(test_data) < 10:
            test_logger.warning("Количество тестовых данных недостаточно для достоверного тестирования.")

        correct_predictions = 0
        total_predictions = len(test_data)
        for i, data_point in enumerate(test_data):
            prediction = model.predict(data_point) if hasattr(model, "predict") else None
            true_label = data_point.get("label", None) if isinstance(data_point, dict) else None

            if prediction == true_label:
                correct_predictions += 1

            test_logger.debug(
                f"Тест {i + 1}/{total_predictions}: Предсказание={prediction}, Истинное значение={true_label}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        results = {"accuracy": accuracy, "correct": correct_predictions, "total": total_predictions}

        test_logger.info(f"Тестирование завершено. Точность={accuracy:.2f}")
        return results
    except Exception as e:
        test_logger.critical(f"Критическая ошибка тестирования: {e}")
        return None
