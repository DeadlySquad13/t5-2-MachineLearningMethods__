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
# Loading extension for reloading editable packages (pip install -e .)
# %load_ext autoreload

# %%
# Reloading editable packages.
# %autoreload
from charts.main import get_metrics_grouped_bar_chart

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
CategoryName = str

# For seeing results of each category for each classifier.
MetricsDataPerCategory = dict[ClassifierName, dict[CategoryName, float]] 
metrics_data_per_category: MetricsDataPerCategory = {}

# For seeing general results of each classifier for each model (category
# metrics are generalized then).
MetricsDataPerModel = dict[ClassifierName, dict[ModelName, float]] 
metrics_data_per_model: MetricsDataPerModel = {}

# %%
def add_metrics_data(classier_name: ClassifierName, model_name: ModelName, results: pd.DataFrame):
    if not metrics_data_per_model.get(classier_name):
        metrics_data_per_model[classier_name] = {}

    metrics_data_per_model[classier_name][model_name] = np.mean(results['Точность'])

    results_per_category = dict(zip(results["Категория"], results["Точность"]))
    metrics_data_per_category[classier_name] = results_per_category

    return metrics_data_per_model, metrics_data_per_category

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
"""
#### Сравнение результатов классификаторов для каждой категории
"""

# %%
def show_classifiers_bar_chart(metrics_data_per_category: MetricsDataPerCategory):
    classifiers_bar_chart = get_metrics_grouped_bar_chart(metrics_data_per_category)
    classifiers_bar_chart["plt"].title('Сравнение классификаторов для каждой категории')
    classifiers_bar_chart["plt"].xlabel('Категория')
    classifiers_bar_chart["plt"].ylabel('Значение метрики')

    classifiers_bar_chart["ax"].legend(title='Классификатор', bbox_to_anchor=(1.2, 1))

    classifiers_bar_chart["plt"].show()

# %%
show_classifiers_bar_chart(metrics_data_per_category)

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
#### Сравнение результатов классификаторов для каждой категории
"""

# %%
show_classifiers_bar_chart(metrics_data_per_category)

# %% [markdown]
"""
### Сравнение результатов моделей и классификаторов
"""

# %%
models_bar_chart = get_metrics_grouped_bar_chart(metrics_data_per_model)
models_bar_chart["plt"].title('Сравнение моделей и классификаторов')
models_bar_chart["plt"].xlabel('Модель')
models_bar_chart["plt"].ylabel('Значение (среднее по всем категориям)')

models_bar_chart["ax"].legend(title='Классификатор', bbox_to_anchor=(1.2, 1))

models_bar_chart["plt"].show()

# %% [markdown]
"""
## Вывод
Как видно по графику сравнения моделей и классификаторов, наиболее успешной оказалась связка TFIDF и LogisticRegression.

В общем, значения моделей с классификатором логистистической регрессии выше на 5%, чем у соответствующих моделей с RandomForestClassifier.

При этом значения моделей при одинаковых классификаторах почти не отличаются: и TFIDF, и CountVectorizer демонстрируют хорошие результаты.
"""

