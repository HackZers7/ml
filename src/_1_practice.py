from pathlib import Path  # библиотека для работы с файлами

import numpy as np
import pandas as pd  # библиотека для работы табличными данными
from matplotlib import pyplot as plt  # библиотека для визуализации
from scipy import stats  # библиотека с математическими операциями


def run():
    data_path = Path.cwd() / 'data'
    train_path = data_path / 'train.csv'
    print("data path: ", data_path)
    print("train path: ", train_path)

    train_dataset = pd.read_csv(train_path)
    # Для получения общей информации о датасете можно вызвать метод .info()
    print(train_dataset.info())

    # Еще один удобный способ посмотреть на наши данные - метод .head(), который возвращает первые n (параметр настраивается) строчек из таблицы
    print(train_dataset.head())

    # Реализуем свой способ оценить количество уникальных признаков в колонках
    print(get_unique_values(train_dataset))
    # Теперь мы можем видеть, что на самом деле у нас присутствуют не только целочисленные признаки (PassengerId), 
    # дробные признаки (Age, Fare), текстовые признаки (Name), но и категориальные. Более того некоторые 
    # категориальные признаки, например, Survived, Sex являются бинарными.

    # Давайте проанализируем датасет на пустые значения
    print(get_number_of_missing_values(train_dataset))
    # Как видим, у нас пропущены значения в колонках Age, Cabin и Embarked
    # Рассмотрим влияение пустых значений в колонке Age на метрики mean, median, mode, variance, std, Quartile
    subset = train_dataset['Age']
    print(get_basic_metrics(subset))

    # Как видим пустых значений много, многие метрики не умеют работать с пропущенными значениями.
    original_result = get_metrics(subset)
    print(original_result)

    # Эксперимент №1
    # Заменить пропущенные значения на mean 
    value = get_metrics(subset)['mean']
    filled_subset = fill_empty_values(subset, value)
    print_dicts_comparison(original_result, get_metrics(filled_subset))

    # Эксперимент №2
    # Заменить пропущенные значения на median
    value = get_metrics(subset)['median']
    filled_subset = fill_empty_values(subset, value)
    print_dicts_comparison(original_result, get_metrics(filled_subset))

    # Эксперимент №3
    # Заменить пропущенные значения на mode
    value = get_metrics(subset)['mode']
    filled_subset = fill_empty_values(subset, value)
    print_dicts_comparison(original_result, get_metrics(filled_subset))

    # Изучим выбросы
    # Начнем с подхода IQR
    outlier_mask = get_interquartile_range_mask(subset)
    print(outlier_mask)

    # Посчитать количество выбросов мы можем с помощью обычной суммы
    print(sum(outlier_mask))
    # Мы можем реализовать следующую функцию для анализа работы IQR
    threshold = 0.8
    figure = plot_interquartile_range(subset, threshold)
    figure.show()

    # Подход Z-Score
    threshold = 3
    outlier_mask = get_z_score_mask(subset, threshold)
    print(outlier_mask)
    sum(outlier_mask)
    threshold = 2
    figure = plot_z_score(subset, threshold)
    figure.show()


def get_unique_values(data: pd.DataFrame) -> dict[str, int]:
    """Данная функция должна подсчитывать количество уникальных значений в каждой из колонок"""
    result_dict = {}

    for label, content in data.items():
        unique_content = list(set(content))
        result_dict[label] = len(unique_content)
    return result_dict


def get_number_of_missing_values(data: pd.Series) -> dict[str, int]:
    """Данная функция должна подсчитывать количество пустых ячеек в каждой из колонок"""
    result_dict = {}

    for label, content in data.items():
        result_dict[label] = sum(pd.isna(content))
    return result_dict


def get_basic_metrics(data: pd.Series) -> dict[str, float]:
    """
    Данная функция должна считать метрики `mean`, `median`, `mode`, `variance`, `std`, `Q1`,`Q2`,`Q3`,`Q4`
    и возвращать словарь со значениями
    """
    result_dict = {}

    result_dict['mean'] = np.mean(data)
    result_dict['median'] = np.median(data)
    result_dict['mode'] = stats.mode(data)
    result_dict['variance'] = np.var(data)
    result_dict['STD'] = np.std(data)
    result_dict['Q1'] = np.quantile(data, 0.25)
    result_dict['Q2'] = np.quantile(data, 0.5)
    result_dict['Q3'] = np.quantile(data, 0.75)
    result_dict['Q4'] = np.quantile(data, 1)
    return result_dict


def get_metrics(data: pd.Series) -> dict[str, float]:
    """
    Данная функция должна считать метрики `mean`, `median`, `mode`, `variance`, `std`, `Q1`,`Q2`,`Q3`,`Q4`
    и возвращать словарь со значениями, учитывая пустые значения
    """
    result_dict = {}

    cropped_data = data.dropna()

    result_dict['mean'] = np.mean(cropped_data)
    result_dict['median'] = np.median(cropped_data)
    result_dict['mode'] = stats.mode(cropped_data).mode
    result_dict['variance'] = np.var(cropped_data)
    result_dict['STD'] = np.std(cropped_data)
    result_dict['Q1'] = np.quantile(cropped_data, 0.25)
    result_dict['Q2'] = np.quantile(cropped_data, 0.5)
    result_dict['Q3'] = np.quantile(cropped_data, 0.75)
    result_dict['Q4'] = np.quantile(cropped_data, 1)
    return result_dict


def fill_empty_values(data: pd.Series, value: float) -> pd.Series:
    return data.fillna(value)


def print_dicts_comparison(dict1: dict[str, float], dict2: dict[str, float]) -> None:
    for metric_name in dict1.keys():
        print(f'{metric_name}: {dict1[metric_name]} -> {dict2[metric_name]}')


def get_interquartile_range_mask(data: pd.Series) -> pd.Series:
    """
    Данная функция должна считать квартили, считать верхнюю и нижнюю границу и возвращать маску
    """
    cropped_data = data.dropna()

    quartile_1 = np.quantile(cropped_data, 0.25)
    quartile_3 = np.quantile(cropped_data, 0.75)

    difference_quartile = quartile_3 - quartile_1

    left_bound = quartile_1 - 1.5 * difference_quartile
    right_bound = quartile_3 + 1.5 * difference_quartile

    outliers_mask = (data < left_bound) | (data > right_bound)
    return outliers_mask


def plot_interquartile_range(data: pd.Series, threshold: float) -> plt.Figure:
    """
    Визуализация IQR
    """

    cropped_data = data.dropna()

    quartile_1 = np.quantile(cropped_data, 0.25)
    quartile_3 = np.quantile(cropped_data, 0.75)

    difference_quartile = quartile_3 - quartile_1

    left_bound = quartile_1 - threshold * difference_quartile
    right_bound = quartile_3 + threshold * difference_quartile

    left_data = []
    right_data = []
    normal_data = []
    for value in data:
        if value < left_bound:
            left_data.append(value)
        elif value > right_bound:
            right_data.append(value)
        else:
            normal_data.append(value)

    figure = plt.gcf()
    axe = figure.gca()

    left_x = np.arange(0, len(left_data))
    normal_x = np.arange(len(left_data), len(data) - len(right_data))
    right_x = np.arange(len(data) - len(right_data), len(data))

    if left_data:
        axe.plot(left_x, np.array(left_data))
    if normal_data:
        print(normal_x.shape, np.array(normal_data).shape)
        axe.plot(normal_x, np.array(normal_data))
    if right_data:
        axe.plot(right_x, np.array(right_data))
    return figure


def get_z_score_mask(data: pd.Series, threshold: float) -> pd.Series:
    """
    Данная функция должна считать Z-Score и возвращать маску выбросов
    """

    mean = data.mean()
    std = data.std()

    outliers_mask = (data - mean) / std > threshold
    return outliers_mask


def plot_z_score(data: pd.Series, threshold: float) -> plt.Figure:
    """
    Визуализация Z-Score
    """

    mean = data.mean()
    std = data.std()

    data = (data - mean) / std

    x_values = np.arange(len(data))

    figure = plt.gcf()
    axe = figure.gca()
    axe.plot(x_values, data)
    axe.plot(x_values, np.full(x_values.shape, threshold), 'k-')
    return figure
