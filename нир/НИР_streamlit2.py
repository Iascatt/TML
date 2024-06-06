import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import matplotlib.pyplot as plt

def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('apple_quality.csv', sep=",", nrows=500)
    return data


def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data = data_in.drop_duplicates()
    data["Acidity"] = data.Acidity.astype(float)
    data = data.dropna()
    data = data.drop(columns=["A_id"])
    # Числовые колонки для масштабирования
    scale_cols = data.select_dtypes(["number"]).columns    
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data[new_col_name] = sc1_data[:,i]
    enc_col = data.select_dtypes(["object"]).columns[0]
    le = LabelEncoder()
    new_col_name = enc_col + '_le'
    data[new_col_name] = le.fit_transform(data[enc_col])

    x_train, x_test, y_train, y_test = train_test_split(data[new_cols],data[new_col_name],
                                                         test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test, data[new_cols + [new_col_name]] 

def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, 
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

class MetricLogger:
    
    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
            'alg': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric']==metric)&(self.df['alg']==alg)].index, inplace = True)
        # Добавление нового значения
        temp = [{'metric':metric, 'alg':alg, 'value':value}]
        self.df = pd.concat([self.df, pd.DataFrame(temp)], ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric']==metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values
    
    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5, 
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a,b in zip(pos, array_metric):
            plt.text(0.5, a-0.05, str(round(b,3)), color='white')
        plt.show()    

def clas_train_model(model_name, model, clasMetricLogger):
    model.fit(x_train, y_train)
    # Предсказание значений
    Y_pred = model.predict(x_test)
    # Предсказание вероятности класса "1" для roc auc
    Y_pred_proba_temp = model.predict_proba(x_test)
    Y_pred_proba = Y_pred_proba_temp[:,1]
    
    precision = precision_score(y_test.values, Y_pred)
    recall = recall_score(y_test.values, Y_pred)
    f1 = f1_score(y_test.values, Y_pred)
    roc_auc = roc_auc_score(y_test.values, Y_pred_proba)
    
    clasMetricLogger.add('precision', model_name, precision)
    clasMetricLogger.add('recall', model_name, recall)
    clasMetricLogger.add('f1', model_name, f1)
    clasMetricLogger.add('roc_auc', model_name, roc_auc)

    fig, ax = plt.subplots(ncols=2, figsize=(10,5))    
    draw_roc_curve(y_test.values, Y_pred_proba, ax[0])
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test.values, ax=ax[1],
                      display_labels=['0','1'], 
                      cmap=plt.cm.Blues, normalize='true')
    fig.suptitle(model_name)
    st.pyplot(fig)

st.set_page_config(layout="wide")
st.header('Модели машинного обучения')
data = load_data()
x_train, x_test, y_train, y_test, prep_data = preprocess_data(data)



if st.checkbox('Корреляционная матрица'):

    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(prep_data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)




if st.checkbox('SVC'):

    col1, col2 = st.columns(2)

    with col1:
        c_slider = st.slider('C:', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        gamma_slider = st.slider('Gamma:', min_value=0.0, max_value=10.0, value=0.5, step=0.1)

    with col2:
        model = SVC(C=c_slider, gamma=gamma_slider, probability=True)
        clasMetricLogger = MetricLogger()

        clas_train_model("SVC", model, clasMetricLogger)

if st.checkbox('Случайный лес'):

    col1, col2 = st.columns(2)

    with col1:
        est_slider = st.slider('n_estimators:', min_value=1, max_value=200, value=100, step=1)

    with col2:
        model = RandomForestClassifier(n_estimators=est_slider)
        clasMetricLogger = MetricLogger()

        clas_train_model("RF", model, clasMetricLogger)

if st.checkbox('Метод ближайших соседей'):

    col1, col2 = st.columns(2)


    with col1:

        cv_slider = st.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)
        step_slider = st.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1)

    with col2:

        #Количество записей
        data_len = data.shape[0]
        #Вычислим количество возможных ближайших соседей
        rows_in_one_fold = int(data_len / cv_slider)
        allowed_knn = int(rows_in_one_fold * (cv_slider-1))
        st.write('Количество строк в наборе данных - {}'.format(data_len))
        st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

        # Подбор гиперпараметра
        n_range_list = list(range(1,allowed_knn,step_slider))
        n_range = np.array(n_range_list)
        st.write('Возможные значения соседей - {}'.format(n_range))
        tuned_parameters = [{'n_neighbors': n_range}]

        clf_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=cv_slider, scoring='roc_auc')
        clf_gs.fit(x_train, y_train)

        st.subheader('Оценка качества модели')

        st.write('Лучшее значение параметров - {}'.format(clf_gs.best_params_))

        # Изменение качества на тестовой выборке в зависимости от К-соседей
        fig1 = plt.figure(figsize=(7,5))
        ax = plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
        st.pyplot(fig1)

        clasMetricLogger = MetricLogger()

        clas_train_model(f"KNN_{n_range}", clf_gs, clasMetricLogger)


