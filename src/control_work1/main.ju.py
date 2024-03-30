# %% [markdown]
"""
# Рубежный контроль №1
## Выполнил: Пакало Александр Сергеевич, студент ИУ5-22М
Вариант 13, согласно ему номера задач: 13 и 33 для первой и второй соответственно.
Для моей группы доп. требование: для произвольной колонки данных построить гистограмму
"""

# %% [markdown]
"""
## Подготовка библиотек и данных
"""

# %%
# Loading extension for reloading editable packages (pip install -e .)
# %load_ext autoreload

# %%
# Reloading editable packages.
# %autoreload
# from lab1.main import get_results

import bokeh  # noqa
# %%
import datashader as ds  # noqa
import holoviews as hv  # noqa
import panel as pn  # noqa
from packaging.version import Version  # noqa

min_versions = dict(ds="0.15.1", bokeh="3.2.0", hv="1.16.2", pn="1.2.0")

for lib, ver in min_versions.items():
    v = globals()[lib].__version__
    if Version(v) < Version(ver):
        print("Error: expected {}={}, got {}".format(lib, ver, v))

# %%
hv.extension("bokeh", "matplotlib")

# %%
import pathlib  # noqa

try:
    import pandas as pd

    columns = ["depth", "id", "latitude", "longitude", "mag", "place", "time", "type"]
    path = pathlib.Path("../../data/earthquakes/earthquakes-projected.parq")
    df = pd.read_parquet(path, columns=columns, engine="fastparquet")
    df.head()
except RuntimeError as e:
    print("The data cannot be read: %s" % e)

# %% [markdown]
"""
### Небольшой разведочный анализ данных
"""

# %%
print(df.shape)
df.head()

# %%
# %matplotlib inline

# %%
import hvplot.pandas  # noqa

df.hvplot.scatter(x="longitude", y="latitude")


# %% [markdown]
"""
## Дополнительное требование для группы
Для произвольной колонки данных построить гистограмму.

Построим гистограмму для колонки "mag" (magnitude - магнитуда).
"""

# %%
df["mag"].hvplot.hist()


# %% [markdown]
"""
## Задача №1 (13)
Для набора данных проведите нормализацию для одного (произвольного)
числового признака с использованием функции "обратная зависимость - 1 / X".

Проведём нормализацию для колонки "mag"
"""

# %%
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import scipy.stats as stats  # noqa


# %%
def diagnostic_plots(df: pd.DataFrame, variable: str):
    plt.figure(figsize=(15, 6))
    # гистограмма
    plt.subplot(1, 2, 1)
    df[variable].hist()
    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()


# %%
df["mag_reciprocal"] = 1 / (df["mag"])

# %%
diagnostic_plots(df, "mag_reciprocal")

# %% [markdown]
"""
### Вывод
Как видно, нормализация такой функцией неудачна.
"""

# %% [markdown]
"""
## Задача №2 (33)
Для набора данных проведите процедуру отбора признаков (feature selection).
Используйте метод обертывания (wrapper method),
алгоритм полного перебора (exhaustive feature selection).
"""

# %% [markdown]
"""
### Составим модель регрессии для реализации метода обертывания
Метод обёртывания использует результаты от обучения для выбора лучших признаков.
Поставим перед собой цель предсказать магнитуду. Для этого составим модель регрессии
на основе `RandomForestRegressor`.
"""

# %% [markdown]
"""
Оставим только землетрясения.
"""

# %%
df["type"].unique()

# %%
df.loc[df["type"] == "earthquake"]
df.head()

# %% [markdown]
"""
Так как методы обёртывания довольно затратны с точки зрения времени вычислений,
и алгоритм полного перебора является очень жадным, рассмотрим лишь подмножество данных
исходного набора.
"""

# %%
small_df = df.sample(frac=0.002)
print(small_df.shape)
small_df.head()

# %%
small_df.hvplot.scatter(x="longitude", y="latitude")

# %% [markdown]
"""
`RandomForestRegressor` для себя требует числовые значения, поэтому
немного предобработаем набор данны
Преобразуем колонку даты в совокупность числовых признаков.

Также избавимся от неинтересующих на данном этапе колонок "id" и "place":
первое является неинформативным,
второе - лишь интерпретация местоположения, выраженного в признаках широты и долготы.
"""


# %%
# small_df.set_index("time", inplace=True)
small_df["dt"] = pd.to_datetime(small_df["time"])
small_df.head()

# %%
# День
small_df["day"] = small_df["dt"].dt.day
# Месяц
small_df["month"] = small_df["dt"].dt.month
# Год
small_df["year"] = small_df["dt"].dt.year
# Часы
small_df["hour"] = small_df["dt"].dt.hour
# Минуты
small_df["minute"] = small_df["dt"].dt.minute
small_df.head()

# %% [markdown]
"""
Подготовим выборку для обучения, убрав неинтересующие признаки и уже конвертированные
в совместимый с регрессором формат "time" и "dt".
"""

# %%
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS  # noqa
from sklearn.model_selection import train_test_split  # noqa

X_train, X_test, y_train, y_test = train_test_split(
    small_df.drop(
        columns=["mag", "mag_reciprocal", "id", "place", "time", "dt", "type"]
    ),
    small_df["mag_reciprocal"],
    test_size=0.2,
    random_state=42,
)
print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)

# %%
# Тестовый прогон регрессора
from sklearn.ensemble import RandomForestRegressor  # noqa

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)
reg.predict(X_test)

# %% [markdown]
"""
Возьмём малое подмножество признаков, чтобы уменьшить время обучения модели.
"""

# %%
# Заглушим уведомления о будущих изменениях в sklearn, которые ещё не
# поддержали в библиотеке mlxtend.
import warnings  # noqa

warnings.filterwarnings("ignore", category=FutureWarning)

efs = EFS(reg, min_features=3, max_features=4, scoring="neg_mean_squared_error", cv=10)

efs.fit(X_train, y_train)

print("Лучшая MSE оценка: %.2f" % efs.best_score_ * (-1))
print("Лучшее подмножество признаков:", efs.best_idx_)

# %%
metric_dict = efs.get_metric_dict()

fig = plt.figure(figsize=(30, 20))
k_feat = sorted(metric_dict.keys())
avg = [metric_dict[k]["avg_score"] for k in k_feat]

upper, lower = [], []
for k in k_feat:
    upper.append(metric_dict[k]["avg_score"] + metric_dict[k]["std_dev"])
    lower.append(metric_dict[k]["avg_score"] - metric_dict[k]["std_dev"])

plt.fill_between(k_feat, upper, lower, alpha=0.2, color="blue", lw=1)

plt.plot(k_feat, avg, color="blue", marker="o")
plt.ylabel("Лучшее MSE +/- стандартное отклонение")
plt.xlabel("Количество признаков")
feature_min = len(metric_dict[k_feat[0]]["feature_idx"])
feature_max = len(metric_dict[k_feat[-1]]["feature_idx"])
plt.xticks(k_feat, [str(metric_dict[k]["feature_names"]) for k in k_feat], rotation=90)
plt.show()

# %% [markdown]
"""
### Вывод
Итого, с условием взять из исходного набора данных минимум 3, а максимум 4 признака,
наилучший результат показывают наборы из признаков, имеющих как минимум широту, долготу и глубину.

Это вполне соотносится с реальностью: большинство землетрясений происходят в одном и том же
месте, ввиду географических особенностей области.
"""
