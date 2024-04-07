import numpy as np
import pandas as pd
from scipy import stats


def run():
    dataset = pd.DataFrame({'student': ['A', 'B', 'C', 'D', 'E', 'F'],
                            'math': [70, 78, 90, 87, 84, 86],
                            'science': [90, 94, 79, 86, 84, 83]})

    print(dataset)

    pearson_r = pearson_correlation(np.array([70, 78, 90, 87, 84, 86]), np.array([90, 94, 79, 86, 84, 83]))
    print("Коэффициент корреляции Пирсона:", pearson_r)
    print("Коэффициент ранговой корреляции Спирмена:", spearman_correlation(dataset['math'], dataset['science']))

    tau = kendall_tau(dataset['math'], dataset['science'])
    print("Коэффициент корреляции Кендалла:", tau)


# Коэффициент корреляции Пирсона (r)
def pearson_correlation(x, y):
    # 1. Вычисляем среднее значение наблюдений
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # 2. Считаем числитель (ковариацию)
    cov_xy = sum((x - mean_x) * (y - mean_y)) / n

    # 3. Считаем знаменатель (стандартные отклонения)
    std_x = (sum((x - mean_x) ** 2) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2) / n) ** 0.5

    # Заметим, что в нашей формуле нет деления на n, так как оно просто сокращается
    return cov_xy / (std_x * std_y)


# Коэффициент ранговой корреляции Спирмена (ρ)
def spearman_correlation(x, y):
    num = len(x)

    # 1. Вычисляем ранги наблюдений
    ranks_a = stats.rankdata(x)
    ranks_b = stats.rankdata(y)

    # 2. Считаем числитель (квадратичную разность рангов)
    difference = (ranks_b - ranks_a) ** 2

    # Заметим, что в нашей формуле нет деления на n, так как оно просто сокращается
    return 1 - (6 * sum(difference)) / (num * (num ** 2 - 1))


# Коэффициент ранговой корреляции Кендалла (τ)
def kendall_tau(x, y):
    n = len(x)

    # Вычисляем ранги наблюдений
    x = stats.rankdata(x)
    y = stats.rankdata(y)

    # Подготовим
    concordant = 0  # Число согласованных пар
    discordant = 0  # Число несогласованных пар

    for i in range(n):
        # Перебираем все пары без повторений
        for j in range(i + 1, n):
            # Проверяем согласованных пары
            # 1 вариант через проверку знаков разностей: np.sign(x[j] - x[i]) == np.sign(y[j] - y[i])
            # 2 вариант через полное условие
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                concordant += 1

            # Проверяем несогласованных пары
            # 1 вариант через проверку знаков разностей: np.sign(x[j] - x[i]) != np.sign(y[j] - y[i])
            elif (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                discordant += 1

    return (concordant - discordant) / (n * (n - 1) / 2)
