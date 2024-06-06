# %% [markdown]
"""
# Лабораторная работа по теме "Предобработка текста"
Выполнил: Пакало Александр Сергеевич ИУ5-22М
"""

## Задание
"""
Для произвольного предложения или текста решите следующие задачи:
1. Токенизация.
2. Частеречная разметка.
3. Лемматизация.
4. Выделение (распознавание) именованных сущностей.
5. Разбор предложения.
"""

# %%
import spacy
from spacy.lang.ru import Russian

# Changes display style for spacy.displacy.
IS_JUPYTER = True

# %%
text1 = "Natural Language Toolkit (NLTK) - одна из наиболее старых и известных библиотек. Spacy - на сегодняшний день одна из наиболее развитых библиотек, предназначенных для обработки естественного языка, в том числе ориентированная на русский язык."
# Хорошо подходит для тестирования частей речи (2 пункт) и разбора предложения
# (пункт 5).
text2 = "На косой косе Косой косил траву косой косой."
# Хорошо подходит для тестирования Named Entity Recognition (4 пункт).
text3 = (
    "Москва - столица России, по преданию ее основал князь Юрий Долгорукий в 1147 году."
)
# 
my_text = "Nyandex - отличная компания, ведь в ней (не) работают такие люди как Пвинкович"

CHOSEN_TEXT = my_text

# %% [markdown]
# ### 1. Токенизация

# %%
nlp = spacy.load("ru_core_news_sm")
spacy_text = nlp(CHOSEN_TEXT)

for token in spacy_text:
    print(token)

# %% [markdown]
# ### 2. Частеречная разметка

# %%
import pandas as pd

# You may need reduce or other method for larger texts because it can get slow:
# for each column we iterate over all dataset.
token_text_column = [token.text for token in spacy_text]
token_position_column = [token.pos_ for token in spacy_text]
token_dep_column = [token.dep_ for token in spacy_text]
token_explanation_column = [f"{spacy.explain(position)}, {spacy.explain(dep)}" for position, dep in zip(token_position_column, token_dep_column)]

pd.DataFrame(
    data={
        "text": token_text_column,
        "position": token_position_column,
        "dep": token_dep_column,
        "explanation": token_explanation_column,
    }
)

# %% [markdown]
# ### 3. Лемматизация

# %%
# You may need reduce or other method for larger texts because it can get slow:
# for each column we iterate over all dataset.
token_lemma_column = [token.lemma_ for token in spacy_text]

pd.DataFrame(
    data={
        "text": token_text_column,
        "lemma": token_lemma_column,
    }
)

# %% [markdown]
# ### 4. Выделение (распознавание) именованных сущностей

# %%
from spacy import displacy

displacy.render(spacy_text, style="ent", jupyter=IS_JUPYTER)

# %%
entity_text_column = [entity.text for entity in spacy_text.ents]
entity_label_column = [entity.label_ for entity in spacy_text.ents]
entity_explanation_column = [spacy.explain(entity) for entity in entity_label_column]

pd.DataFrame(
    data={
        "entity": entity_text_column,
        "label": entity_label_column,
        "explanation": entity_explanation_column,
    }
)

# %% [markdown]
# ### 5. Разбор предложения

# %%
from spacy import displacy

displacy.render(spacy_text, style="dep", jupyter=IS_JUPYTER)

# %% [markdown]
"""
## Вывод
Библиотека `spacy` отлично справляется с задачами преобработки и анализа текстов.
С помощью неё мы можем токенизировать текст, определить леммы, выделить именованные сущности
и даже разобрать предложение.
"""
