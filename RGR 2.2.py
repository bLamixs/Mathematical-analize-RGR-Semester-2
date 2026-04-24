import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from numpy.polynomial.legendre import leggauss


# ===================================================
# 1. КВАДРАТУРНЫЕ ФОРМУЛЫ
# ===================================================

def gauss_legendre_quad(f, a, b, n):
    """Квадратура Гаусса-Лежандра"""
    x, w = leggauss(n)
    x_mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w
    return np.sum(w_mapped * f(x_mapped))


def gauss_radau_quad(f, a, b, n):
    """Формула Радо (фиксирован левый конец)"""
    if n == 2:
        x = np.array([-1.0, 1.0 / 3.0])
        w = np.array([0.5, 1.5])
    elif n == 3:
        x = np.array([-1.0, -0.2898979486, 0.6898979486])
        w = np.array([0.2222222222, 0.8333333333, 0.9444444444])
    elif n == 4:
        x = np.array([-1.0, -0.5753189235, 0.1810662711, 0.8228240810])
        w = np.array([0.125, 0.6576886399, 0.7763869377, 0.4409244223])
    elif n == 5:
        x = np.array([-1.0, -0.7204802713, -0.1671808647, 0.4463139727, 0.8857916078])
        w = np.array([0.08, 0.4462078022, 0.6236530459, 0.5627120303, 0.2874271215])
    else:
        x = np.linspace(-1, 1, n)
        x[0] = -1.0
        w = np.ones(n) * 2.0 / n

    x_mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w
    return np.sum(w_mapped * f(x_mapped))


def gauss_lobatto_quad(f, a, b, n):
    """Формула Лобатто (фиксированы оба конца)"""
    if n == 3:
        x = np.array([-1.0, 0.0, 1.0])
        w = np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0])
    elif n == 4:
        x = np.array([-1.0, -1.0 / np.sqrt(5), 1.0 / np.sqrt(5), 1.0])
        w = np.array([1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0])
    elif n == 5:
        x = np.array([-1.0, -0.6546536707, 0.0, 0.6546536707, 1.0])
        w = np.array([0.1, 0.5444444444, 0.7111111111, 0.5444444444, 0.1])
    elif n == 6:
        x = np.array([-1.0, -0.7650553239, -0.2852315165,
                      0.2852315165, 0.7650553239, 1.0])
        w = np.array([0.0666666667, 0.3784749563, 0.5548583770,
                      0.5548583770, 0.3784749563, 0.0666666667])
    elif n == 7:
        x = np.array([-1.0, -0.8302238963, -0.4688487935, 0.0,
                      0.4688487935, 0.8302238963, 1.0])
        w = np.array([0.0476190476, 0.2768260474, 0.4317453812,
                      0.4876190476, 0.4317453812, 0.2768260474, 0.0476190476])
    else:
        x = np.linspace(-1, 1, n)
        w = np.ones(n) * 2.0 / n
        w[0] = w[-1] = 2.0 / (n * (n - 1))

    x_mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w
    return np.sum(w_mapped * f(x_mapped))


def chebyshev_nodes(n, a=-1, b=1):
    """Узлы Чебышёва первого рода на [a,b]"""
    k = np.arange(1, n + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * n))
    return 0.5 * (b - a) * x_cheb + 0.5 * (a + b)


def chebyshev_quad(f, a, b, n):
    """Квадратура Чебышёвского типа (равные веса)"""
    if n > 7:
        raise ValueError("Чебышёвская формула существует только для n <= 7")
    x = chebyshev_nodes(n, -1, 1)
    w = np.full(n, 2.0 / n)
    x_mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w
    return np.sum(w_mapped * f(x_mapped))


def simpson_quad(f, a, b, n_segments):
    """Составной метод Симпсона"""
    if n_segments % 2 != 0:
        n_segments += 1
    h = (b - a) / n_segments
    x = np.linspace(a, b, n_segments + 1)
    y = f(x)
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:n_segments:2]) + 2 * np.sum(y[2:n_segments - 1:2]))


def trapezoidal_quad(f, a, b, n_segments):
    """Составной метод трапеций"""
    h = (b - a) / n_segments
    x = np.linspace(a, b, n_segments + 1)
    y = f(x)
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))


# ===================================================
# 2. ПРАВИЛО РУНГЕ
# ===================================================

def runge_error(I_h, I_kh, p, k=2):
    """Оценка погрешности по правилу Рунге"""
    error = abs(I_h - I_kh) / (k ** p - 1)
    I_corrected = I_h + (I_h - I_kh) / (k ** p - 1)
    return error, I_corrected


def adaptive_gauss(f, a, b, eps, p=4, max_iter=10):
    """Адаптивное интегрирование методом Гаусса"""
    n = 2
    for iteration in range(max_iter):
        I_n = gauss_legendre_quad(f, a, b, n)
        I_2n = gauss_legendre_quad(f, a, b, 2 * n)
        error, I_corrected = runge_error(I_2n, I_n, p, k=2)
        if error < eps:
            return I_corrected, n, error, iteration
        n *= 2
    return I_corrected, n, error, max_iter


# ===================================================
# 3. ТЕСТОВЫЕ ФУНКЦИИ
# ===================================================

def f1(x): return x ** 2


def f2(x): return np.sin(5 * x)


def f3(x): return np.sqrt(x)


def f4(x): return np.exp(-x ** 2)


exact = {
    'x^2': 1.0 / 3.0,
    'sin(5x)': (1 - np.cos(5)) / 5,
    'sqrt(x)': 2.0 / 3.0,
    'exp(-x^2)': np.sqrt(np.pi) / 2 * erf(1)
}

funcs = {
    'x^2': f1,
    'sin(5x)': f2,
    'sqrt(x)': f3,
    'exp(-x^2)': f4
}


# ===================================================
# 4. СБОР ДАННЫХ ДЛЯ ГРАФИКОВ
# ===================================================

def collect_convergence_data(f, f_name, exact_val, max_n=12):
    """Сбор данных для графика сходимости"""
    n_vals = list(range(2, max_n + 1))
    if f_name == 'sin(5x)':
        n_vals = [2, 4, 6, 8, 10, 12]

    results = {'n': n_vals, 'Gauss': [], 'Radau': [], 'Lobatto': [],
               'Chebyshev': [], 'Simpson': [], 'Trapezoidal': []}

    for n in n_vals:
        results['Gauss'].append(abs(gauss_legendre_quad(f, 0, 1, n) - exact_val))

        if n >= 2:
            results['Radau'].append(abs(gauss_radau_quad(f, 0, 1, n) - exact_val))
        else:
            results['Radau'].append(np.nan)

        if n >= 3:
            results['Lobatto'].append(abs(gauss_lobatto_quad(f, 0, 1, n) - exact_val))
        else:
            results['Lobatto'].append(np.nan)

        if n <= 7:
            results['Chebyshev'].append(abs(chebyshev_quad(f, 0, 1, n) - exact_val))
        else:
            results['Chebyshev'].append(np.nan)

        # Симпсон с числом отрезков = 2n (для сопоставимого числа вычислений)
        results['Simpson'].append(abs(simpson_quad(f, 0, 1, max(2, 2 * n)) - exact_val))
        results['Trapezoidal'].append(abs(trapezoidal_quad(f, 0, 1, max(1, n)) - exact_val))

    return results


def collect_workprecision_data(f, exact_val, max_evals=500):
    """Сбор данных для диаграммы точность-трудоёмкость"""
    data = {
        'Gauss': {'errors': [], 'evals': []},
        'Simpson': {'errors': [], 'evals': []},
        'Trapezoidal': {'errors': [], 'evals': []}
    }

    # Гаусс: увеличиваем число узлов n
    for n in range(1, 21):
        I = gauss_legendre_quad(f, 0, 1, n)
        data['Gauss']['errors'].append(abs(I - exact_val))
        data['Gauss']['evals'].append(n)

    # Симпсон: увеличиваем число отрезков
    n_seg = 2
    while n_seg + 1 <= max_evals:
        I = simpson_quad(f, 0, 1, n_seg)
        data['Simpson']['errors'].append(abs(I - exact_val))
        data['Simpson']['evals'].append(n_seg + 1)  # число вычислений = число отрезков + 1
        n_seg *= 2

    # Трапеции: увеличиваем число отрезков
    n_seg = 1
    while n_seg + 1 <= max_evals:
        I = trapezoidal_quad(f, 0, 1, n_seg)
        data['Trapezoidal']['errors'].append(abs(I - exact_val))
        data['Trapezoidal']['evals'].append(n_seg + 1)
        n_seg *= 2

    return data


# ===================================================
# 5. ПОСТРОЕНИЕ ГРАФИКОВ
# ===================================================

def plot_convergence(results, f_name, save_path=None):
    """График сходимости (log-log)"""
    plt.figure(figsize=(10, 7))

    methods = ['Gauss', 'Radau', 'Lobatto', 'Chebyshev', 'Simpson', 'Trapezoidal']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']

    for method, color, marker in zip(methods, colors, markers):
        err = results[method]
        n_vals = results['n']
        valid = [(n_vals[i], err[i]) for i in range(len(n_vals)) if not np.isnan(err[i]) and err[i] > 0]
        if valid:
            n_valid, e_valid = zip(*valid)
            plt.loglog(n_valid, e_valid, marker=marker, color=color, linewidth=1.5, markersize=8, label=method)

    plt.xlabel('Число узлов n', fontsize=12)
    plt.ylabel('Абсолютная погрешность', fontsize=12)
    plt.title(f'Сходимость квадратурных формул для функции {f_name}', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_workprecision(data, f_name, save_path=None):
    """Диаграмма точность-трудоёмкость (log-log)"""
    plt.figure(figsize=(10, 7))

    # Гаусс
    errors_g = data['Gauss']['errors']
    evals_g = data['Gauss']['evals']
    plt.loglog(evals_g, errors_g, 'o-', color='red', linewidth=2, markersize=8, label='Гаусс')

    # Симпсон
    errors_s = data['Simpson']['errors']
    evals_s = data['Simpson']['evals']
    plt.loglog(evals_s, errors_s, 's-', color='purple', linewidth=2, markersize=8, label='Симпсон')

    # Трапеции
    errors_t = data['Trapezoidal']['errors']
    evals_t = data['Trapezoidal']['evals']
    plt.loglog(evals_t, errors_t, '^-', color='brown', linewidth=2, markersize=8, label='Трапеций')

    plt.xlabel('Число вычислений функции', fontsize=12)
    plt.ylabel('Абсолютная погрешность', fontsize=12)
    plt.title(f'Диаграмма точность-трудоёмкость для {f_name}', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_for_function(f_name, f, exact_val):
    """Генерирует оба графика для одной функции"""
    print(f"\n{'=' * 50}")
    print(f"Обработка функции: {f_name}")
    print('=' * 50)

    # График сходимости
    print("Сбор данных для графика сходимости...")
    conv_data = collect_convergence_data(f, f_name, exact_val, max_n=12)
    safe_name = f_name.replace('^', '').replace('(', '').replace(')', '').replace('*', '')
    plot_convergence(conv_data, f_name, save_path=f'convergence_{safe_name}.png')

    # Диаграмма точность-трудоёмкость
    print("Сбор данных для диаграммы точность-трудоёмкость...")
    wp_data = collect_workprecision_data(f, exact_val, max_evals=500)
    plot_workprecision(wp_data, f_name, save_path=f'workprecision_{safe_name}.png')


def demo_runge():
    """Демонстрация правила Рунге"""
    print("\n" + "=" * 50)
    print("Демонстрация правила Рунге для f(x)=e^{-x^2}")
    print("=" * 50)

    f = f4
    exact_val = exact['exp(-x^2)']

    print("\nСравнение истинной погрешности и оценки по Рунге:")
    print(f"{'n':>4} {'2n':>4} {'Истинная погрешность':>20} {'Оценка Рунге':>15} {'Отношение':>10}")
    print("-" * 60)

    for n in [2, 4, 8]:
        I_n = gauss_legendre_quad(f, 0, 1, n)
        I_2n = gauss_legendre_quad(f, 0, 1, 2 * n)
        true_err = abs(I_2n - exact_val)
        runge_err, I_corr = runge_error(I_2n, I_n, p=4, k=2)
        ratio = runge_err / true_err if true_err > 0 else 1
        print(f"{n:4d} {2 * n:4d} {true_err:20.2e} {runge_err:15.2e} {ratio:10.2f}")


def demo_adaptive():
    """Демонстрация адаптивного интегрирования"""
    print("\n" + "=" * 50)
    print("Адаптивное интегрирование методом Гаусса")
    print("=" * 50)

    f = f4
    exact_val = exact['exp(-x^2)']
    eps = 1e-8

    I_adapt, n, err, it = adaptive_gauss(f, 0, 1, eps, p=4)

    print(f"Требуемая точность: ε = {eps}")
    print(f"Достигнутая погрешность: {err:.2e}")
    print(f"Итоговое число узлов: n = {n}")
    print(f"Число итераций: {it}")
    print(f"Вычисленное значение: {I_adapt:.12f}")
    print(f"Точное значение: {exact_val:.12f}")
    print(f"Абсолютная погрешность: {abs(I_adapt - exact_val):.2e}")


# ===================================================
# 6. ОСНОВНОЙ ЗАПУСК
# ===================================================

if __name__ == "__main__":
    # Настройка стиля графиков
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')

    # Генерация графиков для всех функций
    for name, f in funcs.items():
        plot_all_for_function(name, f, exact[name])

    # Демонстрация правила Рунге
    demo_runge()

    # Демонстрация адаптивного интегрирования
    demo_adaptive()

    print("\n" + "=" * 50)
    print("Все графики успешно сохранены!")
    print("=" * 50)