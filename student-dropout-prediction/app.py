from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Загрузка моделей
try:
    model_magistr = joblib.load('models/dec_tree_model_magistr.pkl')
    model_bak_spec = joblib.load('models/dec_tree_model_bak_spec.pkl')
except Exception as e:
    logger.error(f"Ошибка загрузки моделей: {str(e)}")
    raise

# Список признаков (все столбцы, кроме 'Таргет')
FEATURE_COLUMNS = [
    'Приоритет', 'Cумма баллов испытаний', 'БВИ', 'Балл за инд. достижения',
    'Категория конкурса БВИ', 'Контракт', 'Нуждается в общежитии',
    'Иностранный абитуриент (МОН)', 'Пол', 'Прошло лет с окончания уч. заведения',
    'FromEkaterinburg', 'Human Development Index', 'Полных лет на момент поступления',
    'Особая квота', 'Отдельная квота', 'Целевая квота',
    'всероссийская олимпиада школьников (ВОШ)',
    'олимпиада из перечня, утвержденного МОН РФ (ОШ)', 'Заочная', 'Очно-заочная',
    'Военное уч. заведение', 'Высшее', 'Профильная Школа', 'СПО', 'Боевые действия',
    'Инвалиды', 'Квота для иностранных граждан', 'Сироты', 'PostSoviet', 'others',
    'Код направления 1: 10', 'Код направления 1: 11', 'Код направления 1: 27',
    'Код направления 1: 29', 'Код направления 3: 2', 'Код направления 3: 3',
    'Код направления 3: 4'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    logger.debug(f"Получен запрос: {request.method} /predict")
    if request.method == 'POST':
        logger.debug("Обработка POST-запроса")
        education_level = request.form.get('education_level')
        logger.debug(f"Параметры формы: education_level={education_level}")

        if 'file' not in request.files or request.files['file'].filename == '':
            logger.warning("Файл не выбран")
            return render_template('prediction.html', 
                                   error="Пожалуйста, выберите CSV-файл",
                                   show_results=False,
                                   feature_columns=FEATURE_COLUMNS)
        
        file = request.files['file']
        logger.debug(f"Загружен файл: {file.filename}")
        try:
            data = pd.read_csv(file, sep=';')
            logger.debug(f"Прочитан CSV-файл, столбцы: {list(data.columns)}")
            
            # Проверка наличия необходимых столбцов
            missing_cols = [col for col in FEATURE_COLUMNS if col not in data.columns]
            if missing_cols:
                logger.error(f"Отсутствуют столбцы: {missing_cols}")
                return render_template('prediction.html',
                                       error=f"CSV-файл не содержит необходимые столбцы: {missing_cols}",
                                       show_results=False,
                                       feature_columns=FEATURE_COLUMNS)
            
            # Выбор модели
            model = model_magistr if education_level == 'magistr' else model_bak_spec
            
            # Получение вероятностей вместо классов
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]  # Вероятность класса 1
            else:
                predictions = model.predict(data[FEATURE_COLUMNS]).astype(float)
            
            logger.debug(f"Предсказания выполнены, длина: {len(predictions)}")
            
            # Добавление предсказаний как float
            data['prediction'] = predictions.astype(float)
            
            # Генерация HTML-таблицы с форматом для чисел с плавающей точкой
            result_html = data.to_html(classes='table table-striped', index=False, float_format='%.2f')
            
            # Сохранение результатов в CSV с числами с плавающей точкой
            result_csv = data.to_csv(index=False, sep=';', float_format='%.2f')
            with open('results.csv', 'w', encoding='utf-8') as f:
                f.write(result_csv)
            
            return render_template('prediction.html',
                                   result_html=result_html,
                                   show_results=True,
                                   error=None,
                                   feature_columns=FEATURE_COLUMNS)
        
        except Exception as e:
            logger.error(f"Ошибка обработки файла: {str(e)}")
            return render_template('prediction.html',
                                   error=f"Ошибка обработки файла: {str(e)}",
                                   show_results=False,
                                   feature_columns=FEATURE_COLUMNS)
    
    return render_template('prediction.html', 
                           show_results=False, 
                           error=None, 
                           feature_columns=FEATURE_COLUMNS)

@app.route('/download_results')
def download_results():
    try:
        return send_file('results.csv',
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='predictions.csv')
    except Exception as e:
        logger.error(f"Ошибка скачивания файла: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)