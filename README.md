# Прогнозирование отсева студентов с помощью ML

[![Статья на платформе sibac.info](https://img.shields.io/badge/Научная%20статья-Опубликована-brightgreen)](https://sibac.info/studconf/science/ccxiv/379235)
[![Демо на Render](https://img.shields.io/badge/Демо%20приложение-Доступно%20online-blue)](https://student-dropout-w32x.onrender.com)

Исследование эффективности алгоритмов машинного обучения и методов предобработки данных для прогнозирования отсева студентов в образовательных учреждениях.

## 🔍 Ключевые компоненты исследования

### 📊 Методы обработки выбросов
- **DBSCAN** (Density-Based Spatial Clustering)
- **Расстояние Махаланобиса** (Mahalanobis Distance)
- **Capping** (0.05; 0.95)
- **Isolation Forest**
- **Local Outlier Factor** (LOF)

### 🤖 Использованные модели
| Модель                     | Тип алгоритма          |
|----------------------------|------------------------|
| Logistic Regression        | Линейная модель        |
| Decision Tree              | Дерево решений         |
| AdaBoost Classifier        | Ансамблевый метод      |
| CatBoost Classifier        | Гребущий бустинг       |
| HistGradientBoosting       | Гистограммный бустинг  |
| LGBM Classifier            | Градиентный бустинг    |
| XGB Classifier             | Градиентный бустинг    |
| KNN                        | Метод ближайших соседей|
| Random Forest              | Случайный лес          |

## � Результаты экспериментов
- Протестировано **54 комбинации** методов предобработки и моделей
- **Наивысшая достигнутая точность: 82%** 

## 📚 Научная публикация
Итоги исследования опубликованы в сборнике конференции:  
[**"Анализ эффективности алгоритмов машинного обучения для прогнозирования отсева студентов"**](https://sibac.info/studconf/science/ccxiv/379235)  

## 🚀 Демо-приложение
Интерактивный прототип развернут на платформе Render:  
👉 **[https://student-dropout-w32x.onrender.com](https://student-dropout-w32x.onrender.com)**  
👉 **[Ссылка на репозиторий демо-приложения](https://github.com/thaigirlPOTROSHITEL/Student_dropout_vers_forhost)**  

Функционал:
- Загрузка пользовательских данных
- Интерактивное предсказание риска отсева
- Визуализация результатов прогнозирования

## 🛠 Технические требования
Для запуска ноутбуков:
```bash
Python >= 3.8
Библиотеки: pandas, numpy, scikit-learn, xgboost, catboost, lightgbm, matplotlib, seaborn
