# %% [markdown]
"""
# Контрольная работа по теме "Классификация текста"
Выполнил: Пакало Александр Сергеевич ИУ5-22М
"""

## Задание
"""
Необходимо решить задачу классификации текстов на основе любого выбранного Вами датасета (кроме примера, который рассматривался в лекции). Классификация может быть бинарной или многоклассовой. Целевой признак из выбранного Вами датасета может иметь любой физический смысл, примером является задача анализа тональности текста.

Необходимо сформировать два варианта векторизации признаков - на основе CountVectorizer и на основе TfidfVectorizer.

В качестве классификаторов необходимо использовать два классификатора по варианту для Вашей группы: RandomForestClassifier, LogisticRegression
"""

# %%
RANDOM_SEED = 13

# %% [markdown]
"""
## Набор данных
Проведём классификацию текста используя набор данных [BBC News Archive](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive)
"""

# %% [markdown]
# Подготовка переменных для работы с данными

# %%
from pathlib import Path

data_path = Path("../../data")
external_data_path = data_path / "external"
raw_data_path = data_path / "raw"

dataset_filename = "bbc-news-data.zip"

# %% [markdown]
# Разархивирование набора данных

# %%
import os
import shutil

raw_data_path.mkdir(exist_ok=True)

file_path = external_data_path / dataset_filename
raw_data_path = external_data_path / dataset_filename

if not os.path.isfile(raw_data_path):
    shutil.unpack_archive(file_path, extract_dir=raw_data_path)
    # file_path.unlink()  # Remove archive after extracting it.

# %% [markdown]
# Загрузка данных из csv

# %%
import pandas as pd

df = pd.read_csv(raw_data_path, sep="\t")

# %% [markdown]
"""
### Разведочный анализ данных
Ознакомимся немного с данными, с которыми собираемся работать
"""

# %% [markdown]
# Основные характеристики датасета

# %%
df.head()

# %%
df.tail()

# %% [markdown]
# Размер датасета

# %%
num_of_rows, num_of_columns = df.shape
print(f'Размер датасета: {num_of_rows} строк, {num_of_columns} колонок')

# %% [markdown]
# Определение типов

# %%
df.dtypes

# %% [markdown]
# Проверка на наличие пустых значений

# %%
df.isnull().sum()

# %% [markdown]
# Обработки пустых значений не требуется

# %% [markdown]
# Проверка на уникальные значения

# %%
pd.Series(df["category"].unique())

# %% [markdown]
"""
### Подготовка корпуса
"""

# %% [markdown]
# Некоторые колонки имеют неверные типы данных, их следует преобразовать.
# 
# Строки вместо `object` сделаем `string`, а колонку "category" сделаем
# типа `category`.

# %%
df = df.astype({
    "category": "category",
    "filename": "string",
    "title": "string",
    "content": "string",
})
df.dtypes

# %% [markdown]
# Токенизация

# %% [markdown]
# Загрузка модели spacy.

# %%
import spacy

spacy_prefers_gpu = spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# %% [markdown]
# Токенизация текстовых значений набора данных (кроме названия файла)

# %%
from spacy.tokens.doc import Doc

corpus: list[Doc] = []

for text in (df["title"] + df["content"]).values:
    corpus.append(nlp(text))

corpus[:3]

# %%
assert len(corpus) == num_of_rows

# %% [markdown]
"""
## Классификация
"""

# %% [markdown]
"""
### Подготовка данных для классификации
"""

# %% [markdown]
# Заметим, что хоть spacy при печати и выводит текст, на самом деле это объект.
# Наши модели ожидают увидеть строки.

# %%
spacy_text = nlp('training: nlp!')
spacy_text, type(spacy_text), type(spacy_text[0])

# %%
[token.text for token in spacy_text]

# %% [markdown]
# Поэтому преобразуем наш `corpus` в упрощённый формат, совместимый с word2vec.

# %%
str_corpus = [spacy_text.text for spacy_text in corpus]

# %% [markdown]
# Выберем `X` и `y` среди нашего набора данных

# %%
X = str_corpus
y = df["category"].values

# %% [markdown]
# Составим выборки для обучения

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=RANDOM_SEED
)

# %% [markdown]
# ### Составим pipeline

# %%
import numpy as np
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def accuracy_score_for_classes(
    y_true: np.ndarray, 
    y_pred: np.ndarray) -> dict[int, float]:
    """
    Вычисление метрики accuracy для каждого класса
    y_true - истинные значения классов
    y_pred - предсказанные значения классов
    Возвращает словарь: ключ - метка класса, 
    значение - Accuracy для данного класса
    """
    # Для удобства фильтрации сформируем Pandas DataFrame 
    d = {'t': y_true, 'p': y_pred}
    df = pd.DataFrame(data=d)
    # Метки классов
    classes = np.unique(y_true)
    # Результирующий словарь
    res = dict()
    # Перебор меток классов
    for c in classes:
        # отфильтруем данные, которые соответствуют 
        # текущей метке класса в истинных значениях
        temp_data_flt = df[df['t']==c]
        # расчет accuracy для заданной метки класса
        temp_acc = accuracy_score(
            temp_data_flt['t'].values, 
            temp_data_flt['p'].values)
        # сохранение результата в словарь
        res[c] = temp_acc
    return res

def print_accuracy_score_for_classes(
    y_true: np.ndarray, 
    y_pred: np.ndarray):
    """
    Вывод метрики accuracy для каждого класса
    """

    accs = accuracy_score_for_classes(y_true, y_pred)
    results = pd.DataFrame(data={ "Категория": accs.keys(), "Точность": accs.values() })

    display(results)

    return results


# %%
class EmbeddingVectorizer(object):
    '''
    Для текста усредним вектора входящих в него слов
    '''
    def __init__(self, model):
        self.model = model
        self.size = model.vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([np.mean(
            [self.model[w] for w in words if w in self.model] 
            or [np.zeros(self.size)], axis=0)
            for words in X])

# %%
from sklearn.pipeline import Pipeline


def classifier_pipeline(v, c, scaler=None):
    pipeline_steps = [
        ("vectorizer", v), 
    ]

    if scaler:
        pipeline_steps.append(("scaler", scaler))

    pipeline_steps.append(("classifier", c))

    pipeline = Pipeline(pipeline_steps)

    classifier_X_train = X_train
    classifier_y_train = y_train
    classifier_X_test = X_test
    classifier_y_test = y_test

    pipeline.fit(classifier_X_train, classifier_y_train)
    y_pred = pipeline.predict(classifier_X_test)

    return print_accuracy_score_for_classes(classifier_y_test, y_pred)


# %% [markdown]
# ### Проверка результатов

# %%
ClassifierName = str
ModelName = str

metrics_data: dict[ClassifierName, dict[ModelName, float]] = {}

# %%
def add_metrics_data(classier_name: ClassifierName, model_name: ModelName, results: pd.DataFrame):
    if not metrics_data.get(classier_name):
        metrics_data[classier_name] = {}

    metrics_data[classier_name][model_name] = np.mean(results['Точность'])

    return metrics_data

# %% [markdown]
# ### CountVectorizer
# Scaler не нужен (и его даже невозможно применить, ведь CountVectorizer возвращает
# разреженную матрицу)

# %%
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(ngram_range=(1,3))

# %% [markdown]
# #### LogisticRegression

# %%
add_metrics_data("LogisticRegression", "count vectorizer", classifier_pipeline(count_vectorizer, LogisticRegression(C=5.0)))

# %% [markdown]
# #### RandomForestClassifier

# %%
from sklearn.ensemble import RandomForestClassifier

add_metrics_data("RandomForestClassifier", "count vectorizer", classifier_pipeline(count_vectorizer, RandomForestClassifier(max_depth=200, criterion="gini")))

# %% [markdown]
# ### TFIDF
# Scaler не нужен (и его даже невозможно применить, ведь tfidf возвращает
# разреженную матрицу)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1,3))

# %% [markdown]
# #### LogisticRegression

# %%
add_metrics_data("LogisticRegression", "tfidf", classifier_pipeline(tfidf, LogisticRegression(C=5.0)))

# %% [markdown]
# #### RandomForestClassifier

# %%
from sklearn.ensemble import RandomForestClassifier

add_metrics_data("RandomForestClassifier", "tfidf", classifier_pipeline(tfidf, RandomForestClassifier(max_depth=200, criterion="gini")))

# %% [markdown]
"""
### Сравнение результатов
"""

# %%
import matplotlib.pyplot as plt

def grouped_bar_chart(ax, data: dict[str, list[float]],
                      tick_labels, colors=None, total_width=0.8,
             single_width=1):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax: matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: Dict[str, List[float]]
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[ 1, 2,      3      ],
            "y":[ 1, np.nan, 3      ],
            "z":[ 1, 2,      np.nan ],
        }

    tick_labels: list
        Labels which will be displayed under each group.

        Example:
        tick_labels = ['Accuracy', 'F1', 'Precision']

        Note:
        Length should be the same as number of groups.
        

    colors: array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width: float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    # Number of bars per group.
    n_bars = len(data)

    # The width of a single bar.
    bar_width = total_width / n_bars

    tick_locations = np.arange(len(tick_labels))

    # Iterate over all data.
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar.
        tick_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # One bar plot consists of multiple rectangles.
        rects = ax.bar(tick_locations + tick_offset, values, width=bar_width * single_width,
                     color=colors[i % len(colors)],
                     label=name)
        # Add a handle to the last drawn bar, which we'll need for the legend.
        # bars.append(bars[0])

        # Add label (value) above each bar. Format it to 5 fixed numbers after
        # comma.
        ax.bar_label(rects, fmt="%0.5f", padding=3)

    ax.set_xticks(tick_locations, labels=tick_labels)


# %%
def show_metrics_grouped_bar_chart(metrics_data: dict[str, dict[str, float]]):
    """ Creates grouped bar chart for metrics.
    :param metrics_data: a dictionary of metrics and their values for each
        model.
    :type metrics_data: dict[str, dict[str, float]]

        Example:

        metrics_data = {
            'GridSearchCV': { 
                'Accuracy': 1,
                'a': 2,
                'b': 3,
            },
            'RandomSearchCV': {
                'Roc': 2,
                'a': 3,
                'b': 4,
            },
        }

    """
    # plt.figure(figsize=(7,3))
    width = 15
    height = 15
    plt.rcParams['figure.figsize'] = width, height

    fig, ax = plt.subplots()

    # Extracting unique metric names from our data.
    tick_labels = set()
    for model_metrics in metrics_data.values():
        tick_labels.update(list(model_metrics.keys()))

    """ Converting our dictionary to a `fully filled` dictionary of lists:
    data = {
        'GridSearchCV': [ 
            1, # was 'Accuracy': 1,
            2, # was 'a': 2,
            3, # was 'b': 3,
            None, # didn't have value for 'Roc';
        ],
        'RandomSearchCV': [
            None, # didn't have value for 'Accuracy',
            3, # was 'a': 3,
            4, # was 'b': 4,
            2, # was 'Roc': 2;
        ],
    }
    """
    filled_metrics_data = {}

    for model in metrics_data.keys():
        filled_metrics_data[model] = []
        for metric in tick_labels:
            filled_metrics_data[model].append(metrics_data[model].get(metric)
                                              or np.nan)
            
    
    grouped_bar_chart(ax, filled_metrics_data, tick_labels=tick_labels,
                      total_width=.8, single_width=.9, colors=['#aadddd',
                          '#eebbbb', '#ccbbbb', '#77bb77', "#cb9978"])

    plt.title('Сравнение моделей и классификаторов')
    plt.xlabel('Модель')
    plt.ylabel('Значение (среднее по всем категориям)')

    ax.legend(title='Классификатор', bbox_to_anchor=(1.2, 1))

    # plt.yscale('log')
    plt.show()

# %%
show_metrics_grouped_bar_chart(metrics_data)

# %% [markdown]
"""
## Вывод
Как видно по графику сравнения моделей и классификаторов, наиболее успешной оказалась связка TFIDF и LogisticRegression.

В общем, значения моделей с классификатором логистистической регрессии выше на 5%, чем у соответствующих моделей с RandomForestClassifier.

При этом значения моделей при одинаковых классификаторах почти не отличаются: и TFIDF, и CountVectorizer демонстрируют хорошие результаты.
"""

