"""
Модуль с функциями визуализации результатов численного интегрирования
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from .methods import METHODS_INFO
from .test_functions import TEST_FUNCTIONS, f1, I1_exact, f5, I5_exact
from .runge import runge_error


# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

# Цветовая схема
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


def ensure_dir(directory):
    """Создание директории, если она не существует"""
    os.makedirs(directory, exist_ok=True)


def plot_convergence_all_methods(a=0, b=1, save_dir='.'):
    """
    График сходимости всех методов для функции x^2
    """
    print("Построение графика 1: Сходимость всех методов...")

    f = f1
    I_exact = I1_exact(a, b)
    n_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    plt.figure(figsize=(14, 10))

    for i, (method_key, method_info) in enumerate(METHODS_INFO.items()):
        errors = []
        for n in n_values:
            try:
                I_approx = method_info['func'](f, a, b, n)
                error = abs(I_exact - I_approx)
                errors.append(error)
            except:
                errors.append(np.nan)

        plt.loglog(n_values, errors,
                   color=COLORS[i % len(COLORS)],
                   marker=MARKERS[i % len(MARKERS)],
                   label=f"{method_info['name']} (порядок {method_info['order']})",
                   linewidth=2,
                   markersize=8,
                   markevery=1)

    # Добавляем эталонные линии
    n_ref = np.array(n_values)
    for order, style, label in [(1, '--', 'O(1/n)'),
                                 (2, '-.', 'O(1/n²)'),
                                 (4, ':', 'O(1/n⁴)')]:
        ref_error = 1 / n_ref ** order
        plt.loglog(n_ref, ref_error,
                   color='black',
                   linestyle=style,
                   label=label,
                   alpha=0.5,
                   linewidth=1.5)

    plt.xlabel('Число разбиений n')
    plt.ylabel('Абсолютная погрешность')
    plt.title(f'Сходимость квадратурных формул для функции $f(x)=x^2$ на [{a}, {b}]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    # Сохранение
    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'convergence_all_methods.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_smoothness_effect(a=0, b=1, save_dir='.'):
    """
    График влияния гладкости функции на сходимость метода Симпсона
    """
    print("Построение графика 2: Влияние гладкости на сходимость...")

    n_values = [4, 8, 16, 32, 64, 128, 256, 512]
    method = METHODS_INFO['simpson']['func']

    plt.figure(figsize=(14, 10))

    selected_funcs = ['f1', 'f2', 'f3']

    for i, func_key in enumerate(selected_funcs):
        func_info = TEST_FUNCTIONS[func_key]
        f = func_info['func']
        I_exact = func_info['exact'](a, b)

        errors = []
        valid_n = []
        for n in n_values:
            try:
                I_approx = method(f, a, b, n)
                error = abs(I_exact - I_approx)
                errors.append(error)
                valid_n.append(n)
            except:
                continue

        plt.loglog(valid_n, errors,
                   color=COLORS[i],
                   marker=MARKERS[i],
                   label=f"{func_info['name']} ({func_info['type']})",
                   linewidth=2.5,
                   markersize=8,
                   markevery=1)

    # Добавляем эталонные линии
    n_ref = np.array([4, 512])
    for order, style in [(1, '--'), (2, '-.'), (4, ':')]:
        ref_error = 1e-2 / (n_ref / 4) ** order
        plt.loglog(n_ref, ref_error,
                   color='gray',
                   linestyle=style,
                   label=f'O(1/n^{order})' if order == 1 else None,
                   alpha=0.7,
                   linewidth=1.5)

    plt.xlabel('Число разбиений n')
    plt.ylabel('Абсолютная погрешность')
    plt.title('Влияние гладкости функции на сходимость метода Симпсона')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'smoothness_effect.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_work_precision_diagram(a=0, b=1, save_dir='.'):
    """
    Диаграмма точность-трудоёмкость для сравнения эффективности методов
    """
    print("Построение графика 3: Диаграмма точность-трудоёмкость...")

    f = f1
    I_exact = I1_exact(a, b)
    n_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # Выбираем основные методы для сравнения
    selected_methods = ['midpoint', 'trapezoidal', 'simpson', 'three_eighths']

    plt.figure(figsize=(14, 10))

    for i, method_key in enumerate(selected_methods):
        method_info = METHODS_INFO[method_key]

        n_list = []
        error_list = []

        for n in n_values:
            try:
                I_approx = method_info['func'](f, a, b, n)
                error = abs(I_exact - I_approx)
                n_list.append(n)
                error_list.append(error)
            except:
                continue

        plt.loglog(n_list, error_list,
                   color=COLORS[i],
                   marker=MARKERS[i],
                   label=f"{method_info['name']} (порядок {method_info['order']})",
                   linewidth=2.5,
                   markersize=8,
                   markevery=1)

    # Добавляем линии постоянной точности
    eps_values = [1e-2, 1e-4, 1e-6, 1e-8]
    ylim = plt.ylim()
    for eps in eps_values:
        plt.axhline(y=eps, color='gray', linestyle=':', alpha=0.5)
        plt.text(plt.xlim()[1] * 1.1, eps, f' ε={eps:.0e}',
                 verticalalignment='center', fontsize=10)
    plt.ylim(ylim)

    plt.xlabel('Количество разбиений n (трудоёмкость)')
    plt.ylabel('Достигнутая точность')
    plt.title('Диаграмма точность-трудоёмкость для функции $f(x)=x^2$')
    plt.gca().invert_yaxis()  # меньшая погрешность - лучше
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'work_precision_diagram.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_runge_comparison(a=0, b=1, save_dir='.'):
    """
    Сравнение фактической погрешности с оценкой по Рунге
    """
    print("Построение графика 4: Сравнение с оценкой Рунге...")

    f = f1
    I_exact = I1_exact(a, b)
    n_values = [4, 8, 16, 32, 64, 128, 256]

    # Методы для сравнения
    compare_methods = ['midpoint', 'simpson']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, method_key in enumerate(compare_methods):
        method_info = METHODS_INFO[method_key]
        ax = axes[idx]

        actual_errors = []
        runge_errors = []
        valid_n = []

        for n in n_values:
            try:
                # Вычисляем с текущим и удвоенным шагом
                I_n = method_info['func'](f, a, b, n)
                I_2n = method_info['func'](f, a, b, 2 * n)

                # Фактическая погрешность для I_2n
                actual_error = abs(I_exact - I_2n)

                # Оценка по Рунге
                runge_error_val, _ = runge_error(I_2n, I_n, method_info['order'])

                actual_errors.append(actual_error)
                runge_errors.append(runge_error_val)
                valid_n.append(n)
            except:
                continue

        ax.loglog(valid_n, actual_errors, 'o-',
                  color=method_info['color'] if 'color' in method_info else COLORS[0],
                  label='Фактическая погрешность',
                  linewidth=2,
                  markersize=8)

        ax.loglog(valid_n, runge_errors, 's--',
                  color='red',
                  label='Оценка по Рунге',
                  linewidth=2,
                  markersize=8)

        ax.set_xlabel('Число разбиений n')
        ax.set_ylabel('Погрешность')
        ax.set_title(f'{method_info["name"]} (порядок {method_info["order"]})')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

    plt.suptitle('Сравнение фактической погрешности с оценкой по правилу Рунге', y=1.02)
    plt.tight_layout()

    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'runge_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_eps_dependence(a=0, b=1, save_dir='.'):
    """
    График зависимости необходимого числа разбиений от требуемой точности
    """
    print("Построение графика 5: Зависимость n от требуемой точности...")

    f = f1
    I_exact = I1_exact(a, b)

    eps_values = np.logspace(-2, -8, 7)
    selected_methods = ['midpoint', 'trapezoidal', 'simpson']

    plt.figure(figsize=(14, 10))

    for i, method_key in enumerate(selected_methods):
        method_info = METHODS_INFO[method_key]

        n_required = []
        valid_eps = []

        for eps in eps_values:
            n = 2
            max_n = 10000
            found = False

            while n <= max_n:
                try:
                    I_approx = method_info['func'](f, a, b, n)
                    error = abs(I_exact - I_approx)

                    if error < eps:
                        n_required.append(n)
                        valid_eps.append(eps)
                        found = True
                        break
                except:
                    pass

                n *= 2

            if not found:
                n_required.append(max_n)
                valid_eps.append(eps)

        plt.loglog(valid_eps, n_required,
                   color=COLORS[i],
                   marker=MARKERS[i],
                   label=f"{method_info['name']} (порядок {method_info['order']})",
                   linewidth=2.5,
                   markersize=8,
                   markevery=1)

    plt.xlabel('Требуемая точность ε')
    plt.ylabel('Необходимое число разбиений n')
    plt.title('Зависимость количества разбиений от требуемой точности')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.gca().invert_xaxis()  # большая точность - справа
    plt.tight_layout()

    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'eps_dependence.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_polynomial_error(a=0, b=1, save_dir='.'):
    """
    График погрешности для полиномов (демонстрация точности методов)
    """
    print("Построение графика 6: Погрешность для полиномов...")

    f = f5  # полином 3-й степени
    I_exact = I5_exact(a, b)

    n_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    selected_methods = ['midpoint', 'trapezoidal', 'simpson', 'three_eighths']

    plt.figure(figsize=(14, 10))

    for i, method_key in enumerate(selected_methods):
        method_info = METHODS_INFO[method_key]

        errors = []
        valid_n = []

        for n in n_values:
            try:
                I_approx = method_info['func'](f, a, b, n)
                error = abs(I_exact - I_approx)
                if error > 0:  # исключаем нулевые значения для логарифмической шкалы
                    errors.append(error)
                    valid_n.append(n)
            except:
                continue

        plt.semilogy(valid_n, errors,
                     color=COLORS[i],
                     marker=MARKERS[i],
                     label=f"{method_info['name']} (порядок {method_info['order']})",
                     linewidth=2.5,
                     markersize=8,
                     markevery=1)

    plt.axhline(y=np.finfo(float).eps, color='red', linestyle='--',
                label='Машинная точность', alpha=0.7)

    plt.xlabel('Число разбиений n')
    plt.ylabel('Абсолютная погрешность')
    plt.title(f'Погрешность интегрирования полинома {TEST_FUNCTIONS["f5"]["name"]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'polynomial_error.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_comparison_heatmap(a=0, b=1, save_dir='.'):
    """
    Создание сравнительной таблицы результатов в виде heatmap
    """
    print("Построение графика 7: Сравнительная тепловая карта...")

    f = f1
    I_exact = I1_exact(a, b)

    n_values = [4, 8, 16, 32, 64, 128]
    method_keys = list(METHODS_INFO.keys())

    # Матрица погрешностей
    error_matrix = np.zeros((len(method_keys), len(n_values)))

    for i, method_key in enumerate(method_keys):
        method_info = METHODS_INFO[method_key]
        for j, n in enumerate(n_values):
            try:
                I_approx = method_info['func'](f, a, b, n)
                error = abs(I_exact - I_approx)
                error_matrix[i, j] = np.log10(error + 1e-16)
            except:
                error_matrix[i, j] = 0

    # Создаём heatmap
    plt.figure(figsize=(14, 8))

    im = plt.imshow(error_matrix, cmap='viridis_r', aspect='auto',
                    vmin=-16, vmax=0)

    # Настройка осей
    plt.xticks(range(len(n_values)), [str(n) for n in n_values])
    plt.yticks(range(len(method_keys)), [METHODS_INFO[k]['name'] for k in method_keys])

    # Добавляем значения в ячейки
    for i in range(len(method_keys)):
        for j in range(len(n_values)):
            value = error_matrix[i, j]
            if not np.isnan(value):
                color = 'white' if value < -8 else 'black'
                plt.text(j, i, f'{10 ** value:.1e}',
                         ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, label='log10(погрешности)')
    plt.xlabel('Число разбиений n')
    plt.ylabel('Метод интегрирования')
    plt.title('Сравнительная тепловая карта погрешностей для функции $f(x)=x^2$')
    plt.tight_layout()

    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'comparison_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_methods_visualization(a=0, b=1, n=8, save_dir='.'):
    """
    Визуализация работы различных методов интегрирования
    """
    print("Построение графика 8: Визуализация методов...")

    f = f1
    x = np.linspace(a, b, 1000)
    y = f(x)

    # Точки для методов
    h = (b - a) / n
    x_mid = np.linspace(a + h / 2, b - h / 2, n)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # 1. Левые прямоугольники
    ax = axes[0]
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    for i in range(n):
        x_left = a + i * h
        ax.fill([x_left, x_left + h, x_left + h, x_left],
                [0, 0, f(x_left), f(x_left)],
                alpha=0.3, color='red', label='Левые прямоугольники' if i == 0 else "")
    ax.set_title('Левые прямоугольники')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1)

    # 2. Правые прямоугольники
    ax = axes[1]
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    for i in range(n):
        x_right = a + (i + 1) * h
        ax.fill([x_right - h, x_right, x_right, x_right - h],
                [0, 0, f(x_right), f(x_right)],
                alpha=0.3, color='green', label='Правые прямоугольники' if i == 0 else "")
    ax.set_title('Правые прямоугольники')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1)

    # 3. Средние прямоугольники
    ax = axes[2]
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    for i, xm in enumerate(x_mid):
        ax.fill([xm - h / 2, xm + h / 2, xm + h / 2, xm - h / 2],
                [0, 0, f(xm), f(xm)],
                alpha=0.3, color='orange', label='Средние прямоугольники' if i == 0 else "")
    ax.set_title('Средние прямоугольники')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1)

    # 4. Метод трапеций
    ax = axes[3]
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    for i in range(n):
        x_left = a + i * h
        x_right = x_left + h
        ax.fill([x_left, x_right, x_right, x_left],
                [0, 0, f(x_right), f(x_left)],
                alpha=0.3, color='purple', label='Трапеции' if i == 0 else "")
    ax.set_title('Метод трапеций')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1)

    # 5. Метод Симпсона
    ax = axes[4]
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    n_simpson = n if n % 2 == 0 else n + 1
    h_s = (b - a) / n_simpson
    for i in range(0, n_simpson, 2):
        x0 = a + i * h_s
        x2 = x0 + 2 * h_s
        xs = np.linspace(x0, x2, 50)
        # Параболическая интерполяция
        x1 = x0 + h_s
        p0 = f(x0)
        p1 = f(x1)
        p2 = f(x2)
        ys = (p0 * (xs - x1) * (xs - x2) / ((x0 - x1) * (x0 - x2)) +
              p1 * (xs - x0) * (xs - x2) / ((x1 - x0) * (x1 - x2)) +
              p2 * (xs - x0) * (xs - x1) / ((x2 - x0) * (x2 - x1)))
        ax.fill_between(xs, 0, ys, alpha=0.3, color='brown', label='Симпсон' if i == 0 else "")
    ax.set_title('Метод Симпсона')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1)

    # 6. Метод 3/8
    ax = axes[5]
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    n_38 = n + (3 - n % 3) if n % 3 != 0 else n
    h_38 = (b - a) / n_38
    for i in range(0, n_38, 3):
        x0 = a + i * h_38
        x3 = x0 + 3 * h_38
        ax.fill_between([x0, x3], 0, [f(x0), f(x3)], alpha=0.3, color='cyan', label='3/8' if i == 0 else "")
    ax.set_title('Метод 3/8')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1)

    plt.suptitle(f'Визуализация методов численного интегрирования (n={n})',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, 'methods_visualization.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def generate_all_graphs(a=0, b=1, save_dir='graphs'):
    """
    Генерация всех графиков для отчёта
    """
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ ДЛЯ РАСЧЁТНО-ГРАФИЧЕСКОЙ РАБОТЫ")
    print("=" * 60)
    print(f"Параметры интегрирования: a={a}, b={b}")
    print()

    # Создаём директорию для графиков
    ensure_dir(save_dir)

    # График 1: Сходимость всех методов
    plot_convergence_all_methods(a, b, save_dir)

    # График 2: Влияние гладкости
    plot_smoothness_effect(a, b, save_dir)

    # График 3: Диаграмма точность-трудоёмкость
    plot_work_precision_diagram(a, b, save_dir)

    # График 4: Сравнение с оценкой Рунге
    plot_runge_comparison(a, b, save_dir)

    # График 5: Зависимость от точности
    plot_eps_dependence(a, b, save_dir)

    # График 6: Погрешность для полиномов
    plot_polynomial_error(a, b, save_dir)

    # График 7: Сравнительная тепловая карта
    plot_comparison_heatmap(a, b, save_dir)

    # График 8: Визуализация методов
    plot_methods_visualization(a, b, save_dir=save_dir)

    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print(f"Графики сохранены в папке '{save_dir}/'")
    print("=" * 60)