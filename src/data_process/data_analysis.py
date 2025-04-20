from typing import Optional, Dict, Any
from logger import Logger

analysis_logger = Logger(name="data_analysis", file_name="data_analysis.log").get_logger()

def analyze_data(data: Optional[list]) -> Optional[Dict[str, Any]]:
    analysis_logger.info("Начало анализа данных.")
    try:
        if data is None:
            analysis_logger.warning("Получены пустые данные для анализа.")
            return None

        result = {
            "length": len(data),
            "unique_characters": len(set(data))
        }
        analysis_logger.debug(f"Результаты анализа: {result}")

        if result["length"] < 5:
            analysis_logger.warning("Длина данных слишком мала для корректного анализа.")
        analysis_logger.info("Анализ данных завершён успешно.")
        return result
    except Exception as e:
        analysis_logger.error(f"Ошибка анализа данных: {e}")
        return None
