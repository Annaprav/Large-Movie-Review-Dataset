# Large-Movie-Review-Dataset

## Анализ эмоциональной окраски текстовых отзывов о фильмах

## Введение

Данный проект посвящен анализу эмоциональной окраски текстовых отзывов о фильмах с использованием методов машинного обучения. В проекте используются данные с отзывами о фильмах из набора данных IMDb. Мы строим модели для определения эмоциональной окраски отзывов (положительная или отрицательная) и для предсказания рейтингов фильмов на основе текстовых отзывов.

## Используемые библиотеки

В проекте используются следующие библиотеки Python:
- `numpy`: Для работы с числовыми массивами.
- `pandas`: Для работы с табличными данными.
- `seaborn` и `matplotlib`: Для визуализации результатов.
- `re`: Для работы с регулярными выражениями.
- `nltk`: Библиотека для обработки естественного языка (Natural Language Toolkit).
- `sklearn`: Для построения моделей машинного обучения.
- `WordNetLemmatizer`: Для лемматизации текстовых данных.

## Обработка данных

Перед построением моделей данные проходят через несколько этапов предварительной обработки:
1. Удаление знаков препинания и приведение текста к нижнему регистру.
2. Удаление стоп-слов (слов, которые не несут смысловой нагрузки) и лемматизация текста (приведение слов к их базовой форме).

## Модель для определения эмоциональной окраски отзывов

Для определения эмоциональной окраски текстовых отзывов о фильмах используется модель Random Forest Classifier. Мы векторизуем текстовые данные с помощью TF-IDF и обучаем модель на данных тренировочного набора. Затем, оцениваем точность предсказания модели на тестовом наборе.

## Модель для предсказания рейтингов фильмов

Для предсказания рейтингов фильмов на основе текстовых отзывов также используется модель Random Forest Classifier. Мы объединяем признаки, полученные из векторизации текста, с признаком, отражающим эмоциональную окраску отзыва (положительная или отрицательная). Затем, обучаем модель на данных тренировочного набора и оцениваем точность предсказания на тестовом наборе.

## Ввод собственного отзыва

Вы можете ввести свой собственный отзыв о фильме, и модель автоматически определит его эмоциональную окраску (положительная или отрицательная).
