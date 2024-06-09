# %% [markdown]
"""
# Лабораторная работа по теме "Классификация текста"
Выполнил: Пакало Александр Сергеевич ИУ5-22М
"""

## Задание
"""
Для произвольного набора данных, предназначенного для классификации текстов, решите задачу классификации текста двумя способами:

- Способ 1. На основе CountVectorizer или **TfidfVectorizer**.
- Способ 2. На основе моделей **word2vec** или Glove или fastText.

Сравните качество полученных моделей.
Для поиска наборов данных в поисковой системе можно использовать ключевые слова "datasets for text classification".
"""

# %%
# Loading extension for reloading editable packages (pip install -e .)
# %load_ext autoreload

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
# Заметим, что spacy при печати и выводит текст, на самом деле это объект.
# Word2Vec ожидает увидеть строки либо списки.

# %%
spacy_text = nlp('training: nlp!')
spacy_text, type(spacy_text), type(spacy_text[0])

# %%
[token.text for token in spacy_text]

# %% [markdown]
# Поэтому преобразуем наш `corpus` в упрощённый формат, совместимый с word2vec.

# %%
corpus_for_word2vec = [[token.text for token in spacy_text] for spacy_text in corpus]

corpus_for_word2vec[:3]

# %% [markdown]
# ## Модель word2vec
# Список доступных предобученнных моделей

# %%
from gensim.downloader import info

list(info()['models'].keys())

# %% [markdown]
# Загрузка модели

# %%
from gensim.downloader import info, load

# You can safely restart this cell, gensim will download it only once.
# It still takes some time to load, though.
word2vec_google_news_300_model = load("word2vec-google-news-300")

# %% [markdown]
# Небольшая проверка работы модели

# %%
words = ["king", "queen", "man", "woman"]

# %%
from itertools import permutations

PAIRS = 2
[f"Для пары слов {word_pair} схожесть: {word2vec_google_news_300_model.similarity(*word_pair):1.2f}" for word_pair in permutations(words, PAIRS)]

# %% [markdown]
# ### Обучение собственной модели

# %%
corpus[0].text

# %%
from gensim.models import word2vec

model_trained_on_dataset: word2vec.Word2Vec

# %time model_trained_on_dataset = word2vec.Word2Vec(corpus_for_word2vec, workers=8, min_count=10, window=10, sample=1e-3)

# %%
wv = model_trained_on_dataset.wv

# %%
for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")

# %%
wv['the']

# %%
model_trained_on_dataset.wv.most_similar(positive=['the'], topn=5)

# %% [markdown]
"""
## Классификация
"""

# %% [markdown]
"""
### Подготовка данных для классификации
"""

# %% [markdown]
# Выберем `X` и `y` среди нашего набора данных

# %%
X = corpus_for_word2vec
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
# Аналогичные выборки сделаем для моделей, которые в себя принимают список
# строк, а не список токенов.

# %%
X_str = [spacy_text.text for spacy_text in corpus]

# %%
from sklearn.model_selection import train_test_split

X_str_train, X_str_test, y_str_train, y_str_test = train_test_split(
    X_str, y,
    test_size=0.33,
    random_state=RANDOM_SEED
)


# %% [markdown]
# ### Составим pipeline

# %% [markdown]
"""
Подготовим scaler'ы для последующих моделей.

Например, NaiveBayes умеет работать только с неотрицательными числами.
"""

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()


# %% [markdown]
# Составим pipeline

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


def classifier_pipeline(v, c, scaler=None, corpus_already_tokenized=True):
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

    if not corpus_already_tokenized:
        classifier_X_train = X_str_train
        classifier_y_train = y_str_train
        classifier_X_test = X_str_test
        classifier_y_test = y_str_test

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
# ### Протестируем собственно-обученную модель Word2Vec.

# %% [markdown]
# #### LogisticRegression

# %%
add_metrics_data("LogisticRegression", "our w2v", classifier_pipeline(EmbeddingVectorizer(model_trained_on_dataset.wv), LogisticRegression(C=5.0)))

# %% [markdown]
# LogisticRegression с min_max_scaler

# %%
add_metrics_data("LogisticRegression with scaler", "our w2v", classifier_pipeline(EmbeddingVectorizer(model_trained_on_dataset.wv), LogisticRegression(C=5.0), scaler=min_max_scaler))

# %% [markdown]
# #### MultinomialNaiveBayes

# %%
from sklearn.naive_bayes import MultinomialNB

# NB нужны неотрицательные значения.
add_metrics_data("MultinomialNB", "our w2v", classifier_pipeline(EmbeddingVectorizer(model_trained_on_dataset.wv), MultinomialNB(), scaler=min_max_scaler))

# %% [markdown]
# #### KNeighborsClassifier

# %%
from sklearn.neighbors import KNeighborsClassifier

# KNC нужны значения, распределённые по нормальному распределению.
add_metrics_data("KNeighborsClassifier", "our w2v", classifier_pipeline(EmbeddingVectorizer(model_trained_on_dataset.wv), KNeighborsClassifier(n_neighbors=15), scaler=standard_scaler))

# %% [markdown]
# #### DecisionTreeClassifier

# %%
from sklearn.tree import DecisionTreeClassifier

classifier_pipeline(EmbeddingVectorizer(model_trained_on_dataset.wv), DecisionTreeClassifier(max_depth=20, criterion="gini"))

# %%
classifier_pipeline(EmbeddingVectorizer(model_trained_on_dataset.wv), DecisionTreeClassifier(max_depth=60, criterion="gini"))

# %%
add_metrics_data("DecisionTreeClassifier", "our w2v", classifier_pipeline(EmbeddingVectorizer(model_trained_on_dataset.wv), DecisionTreeClassifier(max_depth=200, criterion="gini")))

# %% [markdown]
# ### Протестируем предобученную модель от google Word2Vec.

# %% [markdown]
# #### LogisticRegression

# %%
add_metrics_data("LogisticRegression", "google w2v", classifier_pipeline(EmbeddingVectorizer(word2vec_google_news_300_model), LogisticRegression(C=5.0)))

# %% [markdown]
# LogisticRegression с min_max_scaler

# %%
add_metrics_data("LogisticRegression with scaler", "google w2v", classifier_pipeline(EmbeddingVectorizer(word2vec_google_news_300_model), LogisticRegression(C=5.0), scaler=min_max_scaler))

# %% [markdown]
# #### MultinomialNaiveBayes

# %%
from sklearn.naive_bayes import MultinomialNB

# NB нужны неотрицательные значения.
add_metrics_data("MultinomialNB", "google w2v", classifier_pipeline(EmbeddingVectorizer(word2vec_google_news_300_model), MultinomialNB(), scaler=min_max_scaler))

# %% [markdown]
# #### KNeighborsClassifier

# %%
from sklearn.neighbors import KNeighborsClassifier

# KNC нужны значения, распределённые по нормальному распределению.
add_metrics_data("KNeighborsClassifier", "google w2v", classifier_pipeline(EmbeddingVectorizer(word2vec_google_news_300_model), KNeighborsClassifier(n_neighbors=15), scaler=standard_scaler))

# %% [markdown]
# #### DecisionTreeClassifier

# %%
from sklearn.tree import DecisionTreeClassifier

add_metrics_data("DecisionTreeClassifier", "google w2v", classifier_pipeline(EmbeddingVectorizer(word2vec_google_news_300_model), DecisionTreeClassifier(max_depth=200, criterion="gini")))

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
add_metrics_data("LogisticRegression", "tfidf", classifier_pipeline(tfidf, LogisticRegression(C=5.0), corpus_already_tokenized=False))

# %% [markdown]
# #### MultinomialNaiveBayes

# %%
from sklearn.naive_bayes import MultinomialNB

# NB нужны неотрицательные значения.
add_metrics_data("MultinomialNB", "tfidf", classifier_pipeline(tfidf, MultinomialNB(), corpus_already_tokenized=False))

# %% [markdown]
# #### KNeighborsClassifier

# %%
from sklearn.neighbors import KNeighborsClassifier

# KNC нужны значения, распределённые по нормальному распределению.
add_metrics_data("KNeighborsClassifier", "tfidf", classifier_pipeline(tfidf, KNeighborsClassifier(n_neighbors=15), corpus_already_tokenized=False))

# %% [markdown]
# #### DecisionTreeClassifier

# %%
from sklearn.tree import DecisionTreeClassifier

add_metrics_data("DecisionTreeClassifier", "tfidf", classifier_pipeline(tfidf, DecisionTreeClassifier(max_depth=200, criterion="gini"), corpus_already_tokenized=False))

# %% [markdown]
"""
### Сравнение результатов
"""

# %%
# Reloading editable packages.
# %autoreload
from charts.main import grouped_bar_chart, get_metrics_grouped_bar_chart

models_bar_chart = get_metrics_grouped_bar_chart(metrics_data)
models_bar_chart["plt"].title('Сравнение моделей и классификаторов')
models_bar_chart["plt"].xlabel('Модель')
models_bar_chart["plt"].ylabel('Значение (среднее по всем категориям)')

models_bar_chart["ax"].legend(title='Классификатор', bbox_to_anchor=(1.25, 1))

models_bar_chart["plt"].show()

# %% [markdown]
"""
> У TFIDF нет "LogisticRegression with scaler", потому что scaler не нужен (и его невозможно применить для разреженной матрицы).
"""

# %% [markdown]
"""
## Вывод
Как видно по графику сравнения моделей и классификаторов, наиболее успешной оказалась связка word2vector, предобученная google, и логистической регрессии со scaler'ом.

В среднем, модель от google продемонстрировала самый высокий результат для любого классификатора. Единственное исключение - логистическая регрессия для tfidf. Это комбинация показала
очень хороший результат, который лишь немного отстаёт от лучших значений google word2vector.

Так, видно, что в среднем tfidf достигает показателей лишь на 2-5 процентов хуже google2vector. Учитывая простоту алгоритма, это удивительный результат.

Обученная нами модель word2vec даёт в лучшем варианте 86%, что является вполне приемлемым. Однако в среднем у неё самые плохие результаты среди рассматриваемых моделей.
Это довольно неожиданно, ведь обучение и тренировка на данных из одной предметной области должны были дать лучшее качество.

Пытаясь найти причины, первым приходит на ум недостаток данных: наш набор не очень спефичный и на несколько порядков меньше набора данных google. Но в то же время
объём нашей выборки хоть и составляет несколько тысяч строк, в каждой из них есть ячейка с содержанием полноценной статьи, то есть количество токенов должно отвечать
запросам word2vec.

Вероятно, первопричина кроется в качестве самого обучения: мы мало проводили работы с корпусом токенов. Сразу после разбиения текста с помощью `spacy`
на токены, мы отправили модель обучаться. Вполне возможно, дополнительные оптимизации на уровне предложений и языка поспособствовали бы улучшению качества
модели. Например, устранение стоп-слов или очень частых слов, несущих малый смысл в рамках нашей задачи (союзы, предлоги), из выборки. На это ещё больше
намекает хороший результат TFIDF, ведь он эти оптимизации проводит "автоматически" ввиду особенностей формулы.

Таким образом, мы обучили три модели и сравнили показатели. Каждая из них даёт хорошие результаты в классификации, однако у всех моделей есть большой недостаток -
неумение работать со словами, не присутствующими в выборке. Наименее заметно это в google word2vector, ведь в ней набор данных колоссальный, однако если стоит задача
обрабатывать любые слова, лучше подойдёт принципиально модифицированная модель, например, fast2text.
"""

