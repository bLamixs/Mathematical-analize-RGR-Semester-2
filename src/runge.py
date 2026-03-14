"""
Модуль с реализацией правила Рунге для оценки погрешности
"""

import numpy as np


def runge_error(I_h, I_kh, p, k=2):
    """
    Оценка погрешности по правилу Рунге

    Параметры:
    I_h - приближение с шагом h
    I_kh - приближение с шагом k*h
    p - порядок точности метода
    k - коэффициент увеличения шага (по умолчанию 2)

    Возвращает:
    error - оценка погрешности для I_h
    I_corrected - уточнённое значение интеграла (экстраполяция Ричардсона)
    """
    error = (I_h - I_kh) / (k ** p - 1)
    I_corrected = I_h + error
    return abs(error), I_corrected


def adaptive_integration(f, a, b, method, p, eps=1e-6, max_iter=100):
    """
    Адаптивное интегрирование с контролем точности по правилу Рунге

    Параметры:
    f - подынтегральная функция
    a, b - пределы интегрирования
    method - метод интегрирования
    p - порядок точности метода
    eps - требуемая точность
    max_iter - максимальное число итераций

    Возвращает:
    I_corrected - уточнённое значение интеграла
    n - число разбиений
    error - достигнутая погрешность
    iterations - число итераций
    """
    n = 2

    for iteration in range(max_iter):
        try:
            I_n = method(f, a, b, n)
            I_2n = method(f, a, b, 2 * n)

            error, I_corrected = runge_error(I_2n, I_n, p)

            if error < eps:
                return I_corrected, n, error, iteration + 1

            n *= 2
        except Exception as e:
            # В случае ошибки удваиваем n и пробуем снова
            n *= 2
            continue

    # Если точность не достигнута, возвращаем последний результат
    I_n = method(f, a, b, n)
    I_2n = method(f, a, b, 2 * n)
    error, I_corrected = runge_error(I_2n, I_n, p)
    return I_corrected, n, error, max_iter


def estimate_order(f, a, b, method, n_values=None):
    """
    Оценка фактического порядка точности метода

    Параметры:
    f - подынтегральная функция
    a, b - пределы интегрирования
    method - метод интегрирования
    n_values - список чисел разбиений (по умолчанию [4, 8, 16])

    Возвращает:
    float - оценка порядка точности
    """
    if n_values is None:
        n_values = [4, 8, 16]

    # Сортируем значения по убыванию, чтобы самое мелкое разбиение было последним
    n_values = sorted(n_values)

    # Вычисляем приближения для всех шагов
    approximations = []
    for n in n_values:
        try:
            I_approx = method(f, a, b, n)
            approximations.append(I_approx)
        except Exception as e:
            print(f"Ошибка при вычислении с n={n}: {e}")
            approximations.append(None)

    # Удаляем неудачные вычисления
    valid_data = [(n, approx) for n, approx in zip(n_values, approximations)
                  if approx is not None]

    if len(valid_data) < 2:
        print("Недостаточно данных для оценки порядка точности")
        return 0.0

    # Берём самое мелкое разбиение как "точное" значение
    n_fine, I_fine = valid_data[-1]

    # Вычисляем погрешности для остальных разбиений
    errors = []
    n_coarse_list = []

    for n_coarse, I_coarse in valid_data[:-1]:
        if n_coarse < n_fine:
            error = abs(I_fine - I_coarse)
            if error > 0:  # Исключаем нулевые погрешности
                errors.append(error)
                n_coarse_list.append(n_coarse)

    if len(errors) >= 2:
        # Оценка порядка по двум последним значениям
        # Используем формулу: p = log(error1/error2) / log(n2/n1)
        # где n2 > n1 (более мелкий шаг)
        if len(errors) >= 2:
            # Берём две последние погрешности (соответствуют самым мелким шагам)
            error_prev = errors[-2]
            error_last = errors[-1]
            n_prev = n_coarse_list[-2]
            n_last = n_coarse_list[-1]

            # Вычисляем отношение шагов и погрешностей
            ratio_n = n_last / n_prev  # должно быть > 1
            ratio_error = error_prev / error_last  # должно быть > 1

            if ratio_n > 1 and ratio_error > 0:
                p = np.log(ratio_error) / np.log(ratio_n)
                return p

    # Если не удалось оценить порядок, возвращаем теоретический
    # (для совместимости с методами, которые дают точные значения)
    return 2.0  # Возвращаем порядок по умолчанию


def estimate_order_robust(f, a, b, method, n_values=None):
    """
    Улучшенная оценка фактического порядка точности метода
    с использованием нескольких пар значений

    Параметры:
    f - подынтегральная функция
    a, b - пределы интегрирования
    method - метод интегрирования
    n_values - список чисел разбиений

    Возвращает:
    float - усреднённая оценка порядка точности
    """
    if n_values is None:
        n_values = [4, 8, 16, 32, 64]

    # Вычисляем приближения
    approximations = {}
    for n in n_values:
        try:
            approximations[n] = method(f, a, b, n)
        except:
            approximations[n] = None

    # Фильтруем успешные вычисления
    valid_n = [n for n in n_values if approximations[n] is not None]

    if len(valid_n) < 3:
        return 0.0

    # Используем самое мелкое разбиение как опорное
    n_fine = max(valid_n)
    I_fine = approximations[n_fine]

    # Оцениваем порядок для разных пар
    orders = []

    for n in valid_n[:-1]:  # все кроме самого мелкого
        if n < n_fine:
            try:
                I_coarse = approximations[n]
                error = abs(I_fine - I_coarse)
                if error > 0:
                    # p = log(error_const) / log(h_ratio)
                    # где error_const = error / (fine_value * h_coarse^p)
                    # Упрощённо: p ≈ log(error_coarse/error_fine) / log(n_fine/n)
                    # Но у нас нет error_fine, поэтому используем приближение
                    h_ratio = n_fine / n
                    # Предполагаем, что error ~ C * h^p
                    # Тогда p ≈ log(error) / log(1/h_ratio)
                    p_est = np.log(error) / np.log(1/h_ratio)
                    orders.append(p_est)
            except:
                continue

    if orders:
        return np.mean(orders)
    return 0.0