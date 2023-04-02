import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Preprocessing:
    def define_types(df):
        # - 0 a 3 - Madrugada
        # - 4 a 7 - Início da manhã
        # - 8 a 11 - Manhã
        # - 12 a 15 - Início da tarde
        # - 16 a 19 - Tarde
        # - 20 a 23 - Noite
        df['CÓDIGO NCM/SH'] = df['CÓDIGO NCM/SH'].astype(str)
        
        df['VALOR NOTA FISCAL'] = df['VALOR NOTA FISCAL'].replace(',','.',regex=True).astype(float)
        df['VALOR UNITÁRIO'] = df['VALOR UNITÁRIO'].replace(',','.',regex=True).astype(float)
        df['VALOR TOTAL'] = df['VALOR TOTAL'].replace(',','.',regex=True).astype(float)
        df['QUANTIDADE'] = df['QUANTIDADE'].replace(',','.',regex=True).astype(float)
        df['DATA EMISSÃO'] = pd.to_datetime(df['DATA EMISSÃO'])

        df['DATA EMISSÃO ANO'] = pd.DatetimeIndex(df['DATA EMISSÃO']).year
        df['DATA EMISSÃO DIA'] = pd.DatetimeIndex(df['DATA EMISSÃO']).day
        df['DATA EMISSÃO HORA'] = pd.DatetimeIndex(df['DATA EMISSÃO']).hour

        df['DATA EMISSÃO MES'] = pd.DatetimeIndex(df['DATA EMISSÃO']).month
        month_dict = {
            1:'Janeiro',
            2:'Fevereiro',
            3:'Março',
            4:'Abril',
            5:'Maio',
            6:'Junho',
            7:'Julho',
            8:'Agosto',
            9:'Setembro',
            10:'Outubro',
            11:'Novembro',
            12:'Dezembro'}
        df.replace({"DATA EMISSÃO MES": month_dict}, inplace=True)

        df['DATA EMISSÃO PERIODO'] = (df['DATA EMISSÃO HORA'] % 24 + 4) // 4
        df['DATA EMISSÃO PERIODO'].replace({1: 'Madrugada',
                            2: 'Início da manhã',
                            3: 'Manhã',
                            4: 'Início da tarde',
                            5: 'Tarde',
                            6: 'Noite'}, inplace=True)

        df['DATA EMISSÃO DIA SEMANA'] = pd.DatetimeIndex(df['DATA EMISSÃO']).weekday
        days_dict = {
            0:'Segunda-feira',
            1:'Terça-feira',
            2:'Quarta-feira',
            3:'Quinta-Feira',
            4:'Sexta-feira',
            5:'Sábado',
            6:'Domingo'}
        df.replace({"DATA EMISSÃO DIA SEMANA": days_dict}, inplace=True)

        return df

    def filter_event_authorized(df):
        filtered_df = df[df['EVENTO MAIS RECENTE'] == 'Autorização de Uso']
        return filtered_df

    def apply_normalization(X):
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalizado = scaler.fit_transform(X)
        return X_normalizado
    