# Промышленное машинное обучение, МФТИ
## Лабораторная работа 4
Ярошенко Артём Б06-807а

### Цель работы:

Получить навыки организации взаимодействия различных моделей
машинного обучения и реализации единого жизненного цикла эксплуатации
(pipeline).

### Ход работы:

#### Создания docker контейнера с базой данных MySQL

Были написаны dockerfile и docker-compose.yaml для развертывания докер контейнера с запущенной базой данных MySQL.
В дальшем база данных использовалась для хранения, выгрузки и загрузки данных моделей машинного обучения использованных
в работах 3 и 4. Докер контейнер получил возможность принимать запорсы из локальной сети по порту 6603.

#### Работа с данными и витрина данных

Аналогично предыдущей работе был реализован класс отвечающий за выгрузку и загрузку данных в базу данных. В классе реализованны исключения, которые могут возникнуть при данных операциях.
Витрина данных возвращает данные в виде pandas.DataFrame которые в дальнейшем встроенными методами превращаются в pySpark.DataFrame.
В данной ветрине происходит также предобработка данных заключающаяся в предварительном разделении их на тестовую и тренировочную выборку методами
библиотеки sklearn.

#### Модель классификации

В качестве модели классификации была выбрана встроенная в pyspark модель наивной Баесовской классификации. Для реализации был написан скрипт
содержащий в себе класс отвечающий за обучение модели и передачу результатов модели на витрину данных. 
Полученная модель показало достаточно высокое качество, accuracy score = 0.89

#### Логичтическая регрессия и оценка точности кластеризации

Был разработан скрипт содержащий в себе класс отвечающий за обучение и передачу на витрину данных результатов работы модели машинного обучения.
Данные полученные в результате применения полученной модели к центрам кластеров кластеризованных данных показали что кластеризация проведена с высокой степенью эффективности.
Это можно объяснить урезанным в 3ей лабораторной работе набором и большим количеством обрабатываемых признаков.

### Выводы

В результате удалось релизовать жизненный цикл модели машинного обучения. При этом была изучена реализованная модель кластеризации методами
регрессионного анализа.