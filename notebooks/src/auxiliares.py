import pandas as pd


def dataframe_coeficientes(coefs, colunas):
    return pd.DataFrame(data=coefs, index=colunas, columns=["coeficiente"]).sort_values(
        by="coeficiente"
    )

def remover_outliers(df, variaveis, qt_inf=0.05, qt_sup=0.95):

    df_filtrado = df.copy()

    for variavel in variaveis:

        lm_inf = df[variavel].quantile(qt_inf)
        lm_sup = df[variavel].quantile(qt_sup)

        df_filtrado = df[(df[variavel] >= lm_inf) & (df[variavel] <= lm_sup)]

    return df_filtrado