import pandas as pd
from logger import Logger

data_logger = Logger(name="data_processing", file_name="data_processing.log").get_logger()


def read_csv(file_path: str) -> pd.DataFrame | None:
    data_logger.info(f"Попытка загрузить данные из файла: {file_path}")
    try:
        data = pd.read_csv(file_path)
        data_logger.info(f"Данные успешно загружены. Количество строк: {len(data)}, столбцов: {len(data.columns)}.")
        return data
    except FileNotFoundError:
        data_logger.error(f"Файл не найден: {file_path}.")
    except pd.errors.EmptyDataError:
        data_logger.warning(f"Файл пуст: {file_path}.")
    except Exception as e:
        data_logger.critical(f"Критическая ошибка при чтении файла: {e}")
    return None


def validate_data(data: pd.DataFrame) -> bool:
    if data is None:
        data_logger.error("Валидация данных невозможна: входные данные отсутствуют.")
        return False

    try:
        if data.empty:
            data_logger.warning("DataFrame пуст.")
            return False

        data_logger.info(f"Данные прошли проверку: строк - {len(data)}, столбцов - {len(data.columns)}.")
        return True
    except Exception as e:
        data_logger.critical(f"Ошибка при проверке данных: {e}")
        return False


def process_data(data: pd.DataFrame) -> pd.DataFrame | None:
    if not validate_data(data):
        return None

    try:
        data_logger.info("Начало обработки данных.")

        initial_len = len(data)
        data = data.drop_duplicates()
        data_logger.debug(f"Удалено {initial_len - len(data)} дубликатов.")

        initial_len = len(data)
        data = data.dropna()
        data_logger.debug(f"Удалено {initial_len - len(data)} строк с пропущенными значениями.")

        data_logger.info("Обработка данных завершена успешно.")
        return data
    except Exception as e:
        data_logger.critical(f"Критическая ошибка при обработке данных: {e}")
        return None
