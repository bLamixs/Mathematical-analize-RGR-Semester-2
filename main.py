#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Основной скрипт для проведения численных экспериментов
и генерации графиков для расчётно-графической работы
"""

import os
import sys

# Добавляем путь к src в системный путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualization import generate_all_graphs
from src.methods import METHODS_INFO
from src.test_functions import TEST_FUNCTIONS, get_function_by_name
from src.runge import adaptive_integration, estimate_order_robust


def print_header(text):
    """Вывод заголовка"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def demo_adaptive_integration():
    """Демонстрация адаптивного интегрирования"""
    print_header("ДЕМОНСТРАЦИЯ АДАПТИВНОГО ИНТЕГРИРОВАНИЯ")

    a, b = 0, 1
    f_info = get_function_by_name('f1')
    f = f_info['func']
    I_exact = f_info['exact'](a, b)

    eps_values = [1e-3, 1e-4, 1e-5, 1e-6]

    print(f"\nФункция: {f_info['name']}")
    print(f"Точное значение: {I_exact:.8f}")
    print("-" * 70)
    print(f"{'ε':<12} {'Метод':<20} {'n':<8} {'Погрешность':<15} {'Итераций':<10}")
    print("-" * 70)

    for eps in eps_values:
        for method_key in ['midpoint', 'simpson']:
            method_info = METHODS_INFO[method_key]
            I_corrected, n, error, iterations = adaptive_integration(
                f, a, b, method_info['func'], method_info['order'], eps
            )
            actual_error = abs(I_exact - I_corrected)
            print(f"{eps:<12.0e} {method_info['name']:<20} {n:<8} {actual_error:<15.2e} {iterations:<10}")


def demo_order_estimation():
    """Демонстрация оценки порядка точности"""
    print_header("ОЦЕНКА ФАКТИЧЕСКОГО ПОРЯДКА ТОЧНОСТИ")

    a, b = 0, 1

    for func_key in ['f1', 'f2', 'f3']:
        f_info = TEST_FUNCTIONS[func_key]
        print(f"\nФункция: {f_info['name']} ({f_info['type']})")
        print("-" * 60)
        print(f"{'Метод':<20} {'Теор. порядок':<15} {'Факт. порядок':<15} {'Примечание':<20}")
        print("-" * 60)

        for method_key in ['midpoint', 'trapezoidal', 'simpson']:
            method_info = METHODS_INFO[method_key]
            try:
                p_estimated = estimate_order_robust(f_info['func'], a, b, method_info['func'])

                # Определяем примечание в зависимости от результата
                note = ""
                if func_key == 'f1' and method_key == 'simpson':
                    note = "(точное значение)"
                elif func_key == 'f3' and method_key == 'simpson':
                    note = "(снижение из-за особенности)"
                elif abs(p_estimated - method_info['order']) < 0.5:
                    note = "(соответствует теории)"
                else:
                    note = "(отклонение)"

                print(f"{method_info['name']:<20} {method_info['order']:<15} {p_estimated:<15.2f} {note:<20}")
            except Exception as e:
                print(f"{method_info['name']:<20} {method_info['order']:<15} {'Ошибка':<15} {str(e)[:20]:<20}")


def main():
    """Основная функция"""
    print_header("ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ: КВАДРАТУРНЫЕ ФОРМУЛЫ НЬЮТОНА-КОТЕССА")

    # Параметры интегрирования
    a, b = 0, 1

    # Демонстрация адаптивного интегрирования
    demo_adaptive_integration()

    # Оценка порядка точности
    demo_order_estimation()

    # Генерация графиков
    print_header("ГЕНЕРАЦИЯ ГРАФИКОВ")
    generate_all_graphs(a, b, save_dir='graphs')

    print("\n✅ Все эксперименты успешно завершены!")
    print("📊 Графики сохранены в папке 'graphs/'")


if __name__ == "__main__":
    main()