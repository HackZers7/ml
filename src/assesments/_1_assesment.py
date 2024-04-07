from pathlib import Path

import numpy as np
import pandas as pd


def run():
    # Загружаем датасет
    data_path = Path.cwd() / 'data'
    train_path = data_path / 'StudentsPerformance.csv'
    train_dataset = pd.read_csv(train_path)

    math_score = train_dataset['math score']
    reading_score = train_dataset['reading score']
    writing_score = train_dataset['writing score']

    # Задача 1
    # Получаем метрики
    math_score_metrics = get_metrics(math_score)
    reading_score_metrics = get_metrics(reading_score)
    writing_score_metrics = get_metrics(writing_score)

    # Печатаем словари
    print_dic('math score', math_score_metrics)
    print_dic('reading score', reading_score_metrics)
    print_dic('writing score', writing_score_metrics)

    # Задача 2
    # Найдем пустые значения
    print_dic('math score nulls', get_metrics(find_and_replace_null(math_score)))
    print_dic('reading score nulls', get_metrics(find_and_replace_null(reading_score)))
    print_dic('writing score nulls', get_metrics(find_and_replace_null(writing_score)))

    # Задача 3
    # Найдем отклонение с помощью Z-score
    print('math score')
    print(z_score_replace(1, math_score))
    print('reading score')
    print(z_score_replace(1, reading_score))
    print('writing score')
    print(z_score_replace(1, writing_score))

    # Задача 4
    # Вычисляем корреляцию с помощью алгоритма Пирсона
    c_math_score_reading_score = get_pearson_correlation(math_score, reading_score)
    c_math_score_writing_score = get_pearson_correlation(math_score, writing_score)
    c_reading_score_writing_score = get_pearson_correlation(reading_score, writing_score)

    print("Корреляция между math score и reading score:", c_math_score_reading_score)
    print("Корреляция между math score и writing score:", c_math_score_writing_score)
    print("Корреляция между reading score и writing score:", c_reading_score_writing_score)


def print_dic(str, data: dict):
    print(str)
    for key, value in data.items():
        print(" - " + key + ':', value)


def get_metrics(subset: pd.Series) -> dict[str, float]:
    """Производит подсчет среднего значения, медианы, дисперсии и стандартного отклонения"""
    result_dict = {}

    # среднее (mean)
    result_dict['mean'] = sum(subset) / len(subset)
    # или
    # result_dict['mean'] = np.mean(subset)

    # медиана (median)
    result_dict['median'] = np.median(subset)

    # дисперсия (Variance)
    result_dict['variance'] = np.mean((subset - result_dict['mean']) ** 2)

    # Ищем стандартное отклонение
    # Для этого возьмем корень из дисперсии
    result_dict['STD'] = result_dict['variance'] ** 0.5

    return result_dict


def find_and_replace_null(data: pd.Series) -> pd.Series:
    # Найдем пустые значения
    null_count = sum(pd.isna(data))
    # Пустых значений нет, но допустим они есть, выполним замену одним из способов

    if null_count != 0:
        # Cредним значением
        value = np.mean(data)

        # Медианой
        # value = np.median(subset)

        data = data.fillna(value)

        # Или удалением строк с пустыми значениями
        # data = data.dropna()

    return data


def z_score_replace(threshold, data: pd.Series) -> pd.Series:
    # outliers_mask = (subset - original_statistics['mean']) / original_statistics['S'] > threshold

    mean = np.mean(data)
    std = np.std(data)

    for el in data:
        z_value = (el - mean) / std
        if -1 * threshold > z_value < threshold:
            data.replace(el, mean)

    return data


def get_pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)

    covariance = np.mean((x - mean_x) * (y - mean_y))

    return covariance / (std_x * std_y)
