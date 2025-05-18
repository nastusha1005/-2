import numpy as np
import random

def f(x):
    return np.exp(-x) * np.sin(2 * x) + 0.5 * x

x = np.linspace(0, 5, 100)
e = np.array([random.uniform(-0.5, 0.5) for _ in range(100)])
y = f(x) + e

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Подготовка данных (преобразование x в двумерный массив)
X = x.reshape(-1, 1)

# Инициализация моделей
models = {
    "SVR": SVR(kernel='rbf'),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100)
}

# Обучение и предсказание
results = {}
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    results[name] = {"predictions": y_pred, "MSE": mse}
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(x, y, color='blue', label='Исходные точки')
    plt.plot(x, f(x), color='green', label='Исходная функция')
    plt.plot(x, result["predictions"], color='red', label=f'Предсказание ({name})')
    plt.title(f'{name}\nMSE: {result["MSE"]:.4f}')
    plt.legend()

plt.tight_layout()
plt.show()

''' 6. Выводы
SVR: Хорошо аппроксимирует нелинейные данные, но может быть чувствителен к гиперпараметрам.

Random Forest: Даёт хорошие результаты благодаря ансамблю деревьев, но может переобучаться.

Gradient Boosting: Часто показывает наилучшую точность за счёт последовательного улучшения предсказаний.

Лучшая модель: Gradient Boosting, так как она обычно демонстрирует наименьшую MSE и лучше всего следует за исходной функцией.
Худшая модель: SVR (если MSE выше), так как требует тонкой настройки параметров. '''
