import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from imblearn.over_sampling import SMOTE
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

SEED = 42

class Preprocessing:
    def fix_column_types(df):
        df['NCM'] = df['NCM'].astype(str)  
        df['CST'] = df['CST'].astype(float).astype('Int64').astype(str)
        return df


    def lower_text(text):
        pp_text = text.lower()
        return pp_text


    def remove_table_caracters(text):
        pp_text = re.sub(r'\n', '', text)
        pp_text = re.sub(r'\d+[.\d*]*\:+', '', pp_text)
        return pp_text


    def replace_quantity_values(text):
        pp_text = re.sub(r'\d+[.]*[,]*[\d]* *[ML|ml|GM|gm|GR|gr|kg|Kg|KG|L|l|ML|ml|MM|mm|%|unds|und|un|UNDS|UND|UN]+', ' QUANTITY ', text)
        pp_text = re.sub(r'\d+[x|X]+\d+', ' QUANTITY ', pp_text)
        return pp_text


    def replace_size_values(text):
        pp_text = re.sub(r' [XP|xp|PP|pp|M|m|G|g|GG|gg|XG|xg] ', ' SIZE ', text)
        return pp_text


    def remove_numbers(text):
        pp_text = re.sub(r'\d+',' ', text)
        return pp_text


    def remove_numbers_except_ncm(text):
        pp_text = re.sub(r'[^\[\d\]]\d+','', text)
        return pp_text


    def remove_punctuation(text):
        pp_text = re.sub(r'c/', ' ', text)
        pp_text = re.sub(r'C/', ' ', pp_text)
        pp_text = re.sub(r'[-|,|;|:|.|_|*|"|\'|#|\(|\)|\/|\\|\[|\]]', ' ', pp_text)
        return pp_text


    def tokenize(text):
        pp_text = word_tokenize(text)
        return pp_text


    def remove_stopwords(text):
        pp_text = [word for word in text if word not in stopwords.words('portuguese')]
        return pp_text


    def lemmatize(text):
        wl = WordNetLemmatizer()
        pp_text = [wl.lemmatize(word) for word in text]
        return pp_text

    def apply_preprocessing(df):
        df = df.fillna('')

        pp_desc = []
        corpus_pp_desc = ''

        for text in df['DESCRICAO']:
            pp_text = Preprocessing.lower_text(text)
            pp_text = Preprocessing.replace_quantity_values(pp_text)
            pp_text = Preprocessing.replace_size_values(pp_text)
            pp_text = Preprocessing.remove_numbers(pp_text)
            pp_text = Preprocessing.remove_punctuation(pp_text)
            pp_text = Preprocessing.tokenize(pp_text)
            pp_text = Preprocessing.remove_stopwords(pp_text)
            pp_text = Preprocessing.lemmatize(pp_text)
            pp_desc.append(pp_text)
            corpus_pp_desc += text+''

        pp_df = df.copy(deep=True)
        pp_df['DESCRICAO'] = pp_desc

        return pp_df, corpus_pp_desc


    def define_types(df):
        # - 0 a 3 - Madrugada
        # - 4 a 7 - Início da manhã
        # - 8 a 11 - Manhã
        # - 12 a 15 - Início da tarde
        # - 16 a 19 - Tarde
        # - 20 a 23 - Noite
        df['CÓDIGO NCM/SH'] = df['CÓDIGO NCM/SH'].astype(str)
        df['CFOP'] = df['CFOP'].astype(str)

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
    
    
    def print_percentage(self, X, dataset):
        percentage = len(X[dataset])*100 / (len(X['test']) + len(X['train']) + len(X['val']))
        print(f'{dataset}: {round(percentage)}%')
        return
    

    def split_dataset(self, corpus, columns, label):
        # Train-val-test split
        X = {}
        y = {}

        X['train'], X['test'], y['train'], y['test'] = train_test_split(corpus[columns], corpus[label], test_size=0.2, random_state=SEED)
        X['train'], X['val'], y['train'], y['val'] = train_test_split(X['train'], y['train'], test_size=0.125, random_state=SEED)
        
        self.print_percentage(X,'train')
        self.print_percentage(X,'val')
        self.print_percentage(X,'test')

        df_train = pd.DataFrame()
        df_train[columns] = X['train'][columns]
        df_train['CAPÍTULO NCM'] = y['train']

        df_val = pd.DataFrame()
        df_val[columns] = X['val'][columns]
        df_val['CAPÍTULO NCM'] = y['val']

        df_test = pd.DataFrame()
        df_test[columns] = X['test'][columns]
        df_test['CAPÍTULO NCM'] = y['test']

        return df_train, df_val, df_test
    

    def split_dataset_num(self, corpus, columns, label):
        # Train-val-test split
        X = {}
        y = {}

        X['train'], X['test'], y['train'], y['test'] = train_test_split(corpus[columns], corpus[label], test_size=0.2, random_state=SEED)
        X['train'], X['val'], y['train'], y['val'] = train_test_split(X['train'], y['train'], test_size=0.125, random_state=SEED)
        
        self.print_percentage(X,'train')
        self.print_percentage(X,'val')
        self.print_percentage(X,'test')

        df_train = pd.DataFrame()
        df_train[columns] = X['train'][columns]
        df_train['CAPÍTULO NCM'] = y['train']

        df_val = pd.DataFrame()
        df_val[columns] = X['val'][columns]
        df_val['CAPÍTULO NCM'] = y['val']

        df_test = pd.DataFrame()
        df_test[columns] = X['test'][columns]
        df_test['CAPÍTULO NCM'] = y['test']

        return df_train, df_val, df_test
    

    def get_sequences_details(df_train):
        sequence_length = 0
        max_sequence_length = 0

        for text_array in df_train['DESCRICAO']:
            sequence_length += len(text_array)
            if max_sequence_length < len(text_array):
                max_sequence_length = len(text_array)
            
        mean_sequence_length = sequence_length/len(df_train['DESCRICAO'])

        return mean_sequence_length, max_sequence_length


    def adapt_X_for_input_layer(df_train_text, df_val_text, df_test_text, max_sequence_length):
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(df_train_text)
        vocab_size = len(tokenizer.word_index) + 1

        # train
        X_train_tokens = tokenizer.texts_to_sequences(df_train_text)
        X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_tokens, maxlen=max_sequence_length)

        # validation
        X_val_tokens = tokenizer.texts_to_sequences(df_val_text)
        X_val_padded = keras.preprocessing.sequence.pad_sequences(X_val_tokens, maxlen=max_sequence_length)

        # test
        X_test_tokens = tokenizer.texts_to_sequences(df_test_text)
        X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_tokens, maxlen=max_sequence_length)

        return vocab_size, X_train_padded, X_val_padded, X_test_padded


    def smote(X, y):
        X_resampled, y_resampled = SMOTE(sampling_strategy='not majority', random_state=SEED).fit_resample(X, y)
        # 'not majority': resample all classes but the majority class

        return X_resampled, y_resampled
    

    def adapt_y_for_input_layer(df_train_labels, df_val_labels, df_test_labels):
        y_train_cat = pd.get_dummies(df_train_labels).values
        y_val_cat = pd.get_dummies(df_val_labels).values
        y_test_cat = pd.get_dummies(df_test_labels).values

        return  y_train_cat, y_val_cat, y_test_cat
