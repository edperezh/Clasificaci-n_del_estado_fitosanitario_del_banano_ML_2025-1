"""
dataFunction2.py

Este módulo contiene funciones para la exploración y visualización de datos
relacionados con un estudio de plantas de banano, sus tratamientos y el avance
de enfermedades en días post-infección (dpi).
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import seaborn as sns


def reporte_datos_ausentes(df):
    """
    Muestra un resumen de valores faltantes en el DataFrame y dibuja un mapa de calor
    para visualizar los datos ausentes.

    Parameters:
    df (pd.DataFrame): DataFrame a evaluar.
    """
    faltantes = df.isnull().sum()
    faltantes = faltantes[faltantes > 0]
    if faltantes.empty:
        print("✔ No hay datos ausentes en el DataFrame.")
    else:
        print("❗ Valores faltantes por columna:")
        print(faltantes)
        # Graficar mapa de calor de nulos
        plt.figure(figsize=(8,4))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title("Mapa de calor: datos ausentes")
        plt.show()


def clasesDiferentes(df):
    """
    Imprime las clases únicas encontradas en las columnas 'dpi', 'Sana' y 'Tratamiento',
    con explicaciones de lo que representan los valores en 'Sana'.

    Parameters:
    df (pd.DataFrame): DataFrame con las columnas de interés.
    """
    print("\nClases diferentes en la columna dpi:")
    print(df['dpi'].unique())
    print("\nClases diferentes en la columna Sana:")
    print(df['Sana'].unique())
    print('Donde:\n')
    print('1 es Sana')
    print('0 es Fusarium')
    print('-1 es E_Hidrico')
    print("\nClases diferentes en la columna Tratamiento:")
    print(df['Tratamiento'].unique())
    print()


def pie_Sanas(df):
    """
    Muestra un gráfico de pastel con la distribución de la variable 'Sana'.

    Parameters:
    df (pd.DataFrame): DataFrame con las columnas 'Sana' y 'dpi'.
    """
    ddf = df[['Sana', 'dpi']].groupby('Sana').count()
    ddf = ddf.values.reshape(3,)
    nombres = ['HyS', 'Fusarium', 'Sana']
    colores = ['#830BD9', "#EE6055","#60D394"]
    plt.pie(ddf, labels=nombres, autopct="%0.1f %%", colors=colores)
    plt.axis("equal")
    plt.show()


def pie_Clases(df):
    """
    Muestra un gráfico de pastel con la distribución de la variable 'Tratamiento'.

    Parameters:
    df (pd.DataFrame): DataFrame con las columnas 'Tratamiento' y 'dpi'.
    """
    ddf = df[['Tratamiento', 'dpi']].groupby('Tratamiento').count()
    nombres = ddf.index.values
    ddf = ddf.values.reshape(8,)
    colores = ["#EE6055","#60D394","#AAF683","#FFD97D","#FF9B85", "#0B4CA5", "#6083A4","#AAC683"]
    plt.pie(ddf, labels=nombres, autopct="%0.1f %%", colors=colores)
    plt.axis("equal")
    plt.show()


def bar_dpi(df):
    """
    Muestra un gráfico de barras con la distribución normalizada de plantas según 'dpi'.

    Parameters:
    df (pd.DataFrame): DataFrame con las columnas 'dpi' y 'Tratamiento'.
    """
    ddf = df[['dpi', 'Tratamiento']].groupby('dpi').count()
    index = ddf.index.values
    valores = ddf.values.reshape(-1,)
    n = valores.sum()
    valores = valores/n  # Normaliza los valores
    nombres = ['dpi ' + str(i) for i in index]
    
    plt.bar(nombres, valores, color='skyblue')
    plt.xlabel("DPI")
    plt.ylabel("Cantidad de Plantas")
    plt.title("Distribución de Plantas por DPI")
    plt.xticks(rotation=45)
    plt.show()

def resumen_Clases_dpi(df):
    """
    Calcula e imprime el porcentaje de cada clase de 'Tratamiento' para cada valor de 'dpi'.

    Parameters:
    df (pd.DataFrame): DataFrame con las columnas 'Tratamiento' y 'dpi'.
    """
    total = df.shape[0]  # Total de muestras en el DataFrame
    dpis = sorted(df['dpi'].unique())  # Lista ordenada de todos los valores de DPI
    tratamientos = df['Tratamiento'].unique()  # Clases de tratamiento únicas

    print("\nPorcentaje de cada clase de tratamiento por cada DPI:")
    for tratamiento in tratamientos:
        print(f"\nTratamiento: {tratamiento}")
        # Para cada DPI, filtrar y calcular el porcentaje sobre el total global
        for dpi_val in dpis:
            mask = (df['Tratamiento'] == tratamiento) & (df['dpi'] == dpi_val)
            count = df[mask].shape[0]                 # Número de muestras en esta combinación
            percent = round((count / total) * 100, 1) # Porcentaje redondeado a 1 decimal
            print(f"  - {percent}% en DPI {dpi_val}")


def plot_describe(df_describe):
    """
    Grafica estadísticas descriptivas de firmas espectrales en tres subplots:
      1) media y desviación estándar,
      2) mínimo y máximo,
      3) percentiles 25, 50 y 75.

    Parameters:
    df_describe (pd.DataFrame): Resultado de df.describe(), 
                                con columnas espectrales comenzando en la columna 6.
    """
    # Extraer longitudes de onda (columnas 6 en adelante) como lista de floats
    wavelengths = [float(col) for col in df_describe.columns[5:]]
    # Mapear índices de describe() a sus etiquetas
    idx_map = {1: 'mean', 2: 'std', 3: 'min', 4: '25%', 5: '50%', 6: '75%', 7: 'max'}
    # Agrupaciones de índices a graficar juntos
    groups = [(1, 2), (3, 7), (4, 5, 6)]

    # Usar estilo compatible con Matplotlib ≥3.8
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(nrows=3, figsize=(10, 6))

    for ax, grp in zip(axes, groups):
        for idx in grp:
            series = df_describe.iloc[idx, 5:]  # Valores de la estadística en todas las longitudes
            ax.plot(wavelengths, series, label=idx_map[idx])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def ver_planta_idx(df, idx):
    """
    Grafica la firma espectral de una muestra específica usando su índice.

    Parameters:
    df (pd.DataFrame): DataFrame con datos espectrales; 
                       las longitudes de onda comienzan en la columna 4.
    idx (int): Índice de la fila a graficar.
    """
    wavelengths = list(range(350, 2501))         # Rango de longitudes de onda de 350 a 2500 nm
    spectrum = df.iloc[idx, 3:]                  # Valores de reflectancia desde la 4ª columna
    dpi_val = df.iloc[idx, 0]                    # Valor de DPI (columna 1)
    tratamiento = df.iloc[idx, 2]                # Etiqueta de tratamiento (columna 3)

    plt.style.use('ggplot')                      # Estilo para líneas claras y fondo sutil
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        wavelengths,
        spectrum,
        color='#DD007D',
        linewidth=1.5,
        label=f"DPI: {dpi_val}, Tratamiento: {tratamiento}"
    )
    ax.set_title("Firma espectral por índice de muestra")
    ax.set_xlabel("Longitud de onda (nm)")
    ax.set_ylabel("Reflectancia")
    ax.legend()
    plt.tight_layout()
    plt.show()


def ver_planta_dpi(df, tratamiento, idxs):
    """
    Compara firmas espectrales de varias muestras en distintos DPI.

    Parameters:
    df (pd.DataFrame): DataFrame con datos espectrales; 
                       las longitudes de onda comienzan en la columna 4.
    tratamiento (str): Etiqueta de tratamiento para la leyenda.
    idxs (list[int]): Índices de las filas a graficar, en el mismo orden que los DPI predefinidos.
    """
    wavelengths = list(range(350, 2501))         # Longitudes de onda de 350 a 2500 nm
    dpi_values = [0, 4, 7, 12, 15]                # DPI a comparar
    colors = ['#00D700', '#BBE200', '#FF8E00', '#EA1B00', '#18180D']

    plt.style.use('ggplot')                      # Estilo para líneas limpias
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, dpi_val, color in zip(idxs, dpi_values, colors):
        spectrum = df.iloc[idx, 3:]              # Reflectancia de la muestra
        ax.plot(
            wavelengths,
            spectrum,
            color=color,
            linewidth=1.5,
            label=f"DPI: {dpi_val}, Tratamiento: {tratamiento}"
        )

    ax.set_title("Comparación de firmas espectrales por DPI")
    ax.set_xlabel("Longitud de onda (nm)")
    ax.set_ylabel("Reflectancia")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def ver_mean_std(df, filtro):
    # Filtra el DataFrame según la condición booleana (filtro)
    df_filtro = df[filtro]
    n = df_filtro.shape[0]  # Número total de muestras seleccionadas

    # Lista de longitudes de onda de 350 a 2500 nm
    lgtd_onda = [i for i in range(350, 2501)]

    # Estilo de gráfico
    plt.style.use('seaborn-v0_8')

    # Crear figura con dos subplots: uno para espectros + media, otro para std
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6))

    # Graficar cada muestra en primer subplot
    for i in range(n):
        y = df_filtro.iloc[i, 3:]  # Solo datos espectrales (omitimos metadatos)
        ax0.plot(lgtd_onda, y, color='#99B098')  # Color gris verdoso para cada espectro

    # Calcular y graficar media en rosado grueso
    y = df_filtro.iloc[:, 3:].mean(axis=0).values
    ax0.plot(lgtd_onda, y, color='#CF0079', linewidth=2.5, label='mean')
    ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Calcular y graficar std en turquesa en segundo subplot
    y = df_filtro.iloc[:, 3:].std(axis=0).values
    ax1.plot(lgtd_onda, y, color='#32AFA2', linewidth=2.5, label='std')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Mostrar figura completa
    plt.show()


def explainedVarianceLongitudesOnda(df):
    # Transponer matriz para tener espectros en columnas
    X_ = df.iloc[:, 3:].values.T

    # Calcular matriz de covarianza
    cov_M = np.cov(X_)

    # Obtener valores y vectores propios
    eigen_vals, eigen_vecs = np.linalg.eig(cov_M)

    # Tomar parte real (por seguridad)
    eigen_vals = eigen_vals.real
    tot = sum(eigen_vals).real

    # Ordenar de mayor a menor y normalizar
    A = sorted(eigen_vals, reverse=True)
    var_exp = [(i / tot) for i in A][:10]

    # Calcular varianza acumulada
    cum_var_exp = np.cumsum(var_exp)

    # Preparar x para graficar componentes
    x = [i + 1 for i in range(len(var_exp))]

    # Imprimir detalles en consola
    print(cov_M.shape)
    print('Numero de componentes:', len(eigen_vals))
    print('Primeras 5 componentes de Explained variance ratio')
    s = 0
    for k in range(5):
        print(f'variance ratio {k+1}: {var_exp[k]}')
        s += var_exp[k]
    print(f'Sum explained variance ratio: {s}')

    # Graficar curva de varianza explicada y acumulada
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()
    ax.plot(x, var_exp, color='#DE0079', marker='o', markersize=5, label='Explained variance ratio')
    ax.step(x, cum_var_exp, where='post', color='#00C19B', marker='o', markersize=4, label='Cumulative explained variance')
    ax.set_xlabel('Principal component index', fontsize=12)
    ax.set_ylabel('Explained variance ratio', fontsize=12)
    plt.legend()
    plt.show()


def plot_datos_PCA2d(df):
    # Realiza PCA para reducir dimensiones a 2 componentes principales
    pca = PCA(n_components=2)
    X = df.iloc[:, 3:].values
    X = pca.fit(X).transform(X)

    # Unir componentes con primeros metadatos del DataFrame
    df1 = df.iloc[:, :3]
    df2 = pd.DataFrame(X, columns=['eje_x', 'eje_y'])
    ddf = pd.concat([df1, df2], axis=1)

    # Estilo de gráfico
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # Graficar cada clase con color distinto
    Sana = [0, 1, -1]
    Colores = ['#FF7400', '#3EE600', '#0B29C9']
    Label = ['Fusarium', 'Sana', 'HyS']
    for i in range(len(Sana)):
        filtro = (ddf['Sana'] == Sana[i])
        df_filtro = ddf[filtro]
        x = df_filtro['eje_x'].values
        y = df_filtro['eje_y'].values
        ax.scatter(x, y, color=Colores[i], marker='o', label=Label[i])

    # Agregar leyenda y mostrar gráfico
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_datos_LDA2d(df):
    # Aplica análisis discriminante lineal para reducir a 2 dimensiones
    lda = LinearDiscriminantAnalysis(n_components=2)
    X = df.iloc[:, 3:].values  # Variables espectrales
    y = df['Sana'].values  # Etiquetas de clase
    X = lda.fit(X, y).transform(X)

    # Construir DataFrame transformado con etiquetas
    df_transform = pd.DataFrame(X, columns=['eje_x', 'eje_y'])
    df_transform['Sana'] = y

    plt.style.use('ggplot')  # Estilo de gráfico
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    # Graficar puntos según clase
    Sana = [0, 1, -1]
    Colores = ['#FF7400', '#3EE600', '#0B29C9']
    Label = ['Fusarium', 'Sana', 'HyS']
    for i in range(len(Sana)):
        filtro = (df_transform['Sana'] == Sana[i])
        df_filtro = df_transform[filtro]
        x = df_filtro['eje_x'].values
        y = df_filtro['eje_y'].values
        ax.scatter(x, y, color=Colores[i], marker='o', label=Label[i])

    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()


def plot_datos_tSNE(df):
    # Aplica t-SNE para reducir dimensionalidad a 2D
    X = df.iloc[:, 3:].values
    y = df['Sana'].values
    params = {'n_components':2, 'perplexity':30, 'learning_rate':200, 'n_iter':1500, 'init':'pca', 'random_state':42 }
    X = TSNE(**params).fit_transform(X)

    # Crear DataFrame con resultado de t-SNE y etiquetas
    df_transform = pd.DataFrame(X, columns=['eje_x', 'eje_y'])
    df_transform['Sana'] = y

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    # Graficar cada clase con color distinto
    Sana = [0, 1, -1]
    Colores = ['#FF7400', '#3EE600', '#0B29C9']
    Label = ['Fusarium', 'Sana', 'HyS']
    for i in range(len(Sana)):
        filtro = (df_transform['Sana'] == Sana[i])
        df_filtro = df_transform[filtro]
        x = df_filtro['eje_x'].values
        y = df_filtro['eje_y'].values
        ax.scatter(x, y, color=Colores[i], marker='o', label=Label[i])

    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()


def plot_heat_map(df):
    # Calcula matriz de correlación entre columnas
    data = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='plasma')  # Mapa de calor de correlación

    # Añadir barra de color
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Color bar", rotation=-90, va="bottom")

    plt.show()


def kfoldCV_clf(model, X, y, n_splits):
    # Configura validación cruzada con k-fold
    kfold = KFold(n_splits=n_splits).split(X, y)
    scores = []

    # Iterar por cada fold
    for k, (train, test) in enumerate(kfold):
        model.fit(X[train], y[train])
        score = model.score(X[test], y[test])
        scores.append(score)
        print(f'fold {k+1}', f'accuracy {score:.3f}')

    # Calcular y mostrar media y std de la exactitud
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f'\nCV accuracy: {mean_score:.3f} +/- {std_score:.3f}')


def model_confmat(model, X, y):
    # Calcular exactitud y matriz de confusión
    accuracy = model.score(X, y)
    print(f'\naccuracy: {accuracy:.3f}')

    y_predict = model.predict(X)
    confmat = confusion_matrix(y_true=y, y_pred=y_predict)
    print('\nConfusion Matrix:')
    print(confmat)

    # Dibujar matriz de confusión
    clases = np.unique(y)
    plt.style.use('classic')
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes()
    ax.matshow(confmat, cmap='viridis', alpha=0.3)

    # Escribir valores dentro de la matriz
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')

    # Configurar ejes y etiquetas
    n_labels = [i for i in range(confmat.shape[0])]
    ax.set_xticks(n_labels)
    ax.set_yticks(n_labels)
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('Predict label')
    plt.ylabel('True Label')
    plt.show()

def learningCurve_clf(model, X, y, n_splits):
    # Calcula curva de aprendizaje usando validación cruzada
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, n_splits),  # Proporciones del conjunto de entrenamiento
        cv=n_splits,
        n_jobs=-1,  # Usa todos los núcleos de CPU disponibles
        shuffle=False,
        random_state=33
    )

    # Calcular media y desviación estándar de las métricas de entrenamiento y validación
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Crear figura y graficar resultados
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # Curva de entrenamiento con área sombreada (std)
    ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training acc')
    ax.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')

    # Curva de validación con área sombreada (std)
    ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation acc')
    ax.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

    # Configurar etiquetas y título
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14)

    plt.legend()
    plt.show()


def kfoldCV_reg(model, X, y, n_splits):
    # Configura validación cruzada para regresión
    kfold = KFold(n_splits=n_splits).split(X, y)
    scores = []

    # Iterar por cada fold
    for k, (train, test) in enumerate(kfold):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        score = mean_absolute_error(y_true=y[test], y_pred=y_pred)
        scores.append(score)
        print(f'fold {k+1}', f'mean_absolute_error {score:.3f}')

    # Calcular media y desviación estándar del error
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f'\nCV mean_absolute_error: {mean_score:.3f} +/- {std_score:.3f}')


def plotRegModel(y_pred, y_df, Sanos):
    # Copia el DataFrame y agrega predicciones
    df = y_df.copy()
    df['dpi predict'] = y_pred

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # Si se indica, graficar muestras del tratamiento de control (sanos) por separado
    if Sanos == True:
        filtro = (df['Tratamiento'] == 'Control')
        dff = df[filtro]
        x = dff.index.values
        y_ = dff['dpi predict'].values
        y = dff['dpi'].values
        ax.plot(x, y, label='data Con', marker='x', linestyle='None', color="#028FA3")
        ax.plot(x, y_, label='predict Con', marker='x', linestyle='None', color="#FF8100")

    # Graficar muestras no sanas (enfermas)
    filtro = (df['Tratamiento'] != 'Control')
    dff = df[filtro]
    x = dff.index.values
    y_ = dff['dpi predict']
    y = dff['dpi']

    ax.plot(x, y, label='data', marker='.', linestyle='None', color="#028FA3")
    ax.plot(x, y_, label='predict', marker='.', linestyle='None', color="#FF8100")

    # Configurar etiquetas y título
    ax.set(xlabel='Sample index', ylabel='dpi: dias', title='dpi en test')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def learningCurve_reg(model, X, y, n_splits):
    # Calcula la curva de aprendizaje para un modelo de regresión
    # usando la métrica de error absoluto medio negativo (MAE)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, n_splits),  # Tamaños de conjuntos de entrenamiento
        cv=n_splits,  # Número de folds para validación cruzada
        scoring='neg_mean_absolute_error',  # Métrica de evaluación
        n_jobs=-1,  # Usa todos los núcleos disponibles
        shuffle=False,
        random_state=33
    )

    # Calcular la media y desviación estándar de los errores para cada tamaño de entrenamiento
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Configurar el estilo del gráfico y la figura
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # Graficar curva de entrenamiento con área sombreada (desviación estándar)
    ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training MAE')
    ax.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')

    # Graficar curva de validación con área sombreada
    ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation MAE')
    ax.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

    # Etiquetas y título del gráfico
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('neg_mean_absolute_error', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14)

    # Mostrar leyenda y gráfico
    plt.legend()
    plt.show()

