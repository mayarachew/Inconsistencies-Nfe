# Nfe

10 perguntas para guiar as análises na sua aplicação.

1- Quais os NCM's predominantes neste segmento? [segmento pode ser: conjunto de dados, uma empresa, determinado período, um cluster, etc]
2- Quais os NCM's discrepantes neste segmento? [outliers]
3- A ocorrência deste NCM ou daquele CST neste segmento é homogênea? [cluster bem definido ou não]
NFe
4- Quais os produtos (NCM) que determinada UF mais vende?
5- Quais os produtos (NCM) que determinada UF mais compra?
6- Qual o período do ano que determinado produto (NCM) é mais vendido/comprado?
7- Existe sazonalidade nas operações de compra e venda de determinado NCM?
8- Quais as operações discrepantes para determinada UF em determinado período do ano? [é possível identificar um padrão?]
9- É possível identificar vendas superfaturadas em determinado segmento? [uma venda com valores unitários discrepantes pode ser superfaturada]
10- É possível identificar uma venda discrepante de determinada empresa? [discrepante em termos de valor unitário ou de NCM predominante]

## 1) Questões 

1 - Quais tipos de produto a UF mais compra/vende com o passar dos meses?

- Serie temporal UF DESTINATARIA, mostrar famílias de NCM

- Serie temporal UF EMITENTE, mostrar famílias de NCM

<!-- OU 

- escolher uma UF DESTINATÁRIA, clusterizar por NCM, MES, DIA SEMANA, DIA, UF EMITENTE, aplicar KMODES

- escolher uma UF EMITENTE, clusterizar por NCM, MES, DIA SEMANA, DIA, UF DESTINATÁRIA, aplicar KMODES -->

2 - Existe sazonalidade nas operações de compra e venda de determinado NCM?

- Serie temporal NCM, mostrar famílias de NCM

<!-- - escolher um NCM, clusterizar por MES, DIA DA SEMANA, DIA, PERIODO, UF EMITENTE, UF DESTINATÁRIA, aplicar KMODES -->

3 - É possível identificar vendas superfaturadas em determinado segmento?  [uma venda com valores unitários discrepantes pode ser superfaturada]

<!-- - escolher um conjunto de NCMs de grupos parecidos, clusterizar por MES, DIA SEMANA, DIA, UF DESTINATARIA, UF EMITENTE, INDICADOR IE DESTINATÁRIO, aplicar KMODES, visualizar outlier superior VALOR UNITÁRIO de cada cluster -->

- escolher um conjunto de NCMs de grupos parecidos, clusterizar por VALOR DA NOTA FISCAL, VALOR UNITÁRIO, QUANTIDADE, aplicar MEANSHIFT

4 - É possível identificar uma venda discrepante de determinada empresa? [discrepante em termos de valor unitário ou de NCM predominante]

- escolher uma RAZÃO SOCIAL EMITENTE, clusterizar por NCM, MES, DIA SEMANA, DIA, UF DESTINATARIA, INDICADOR IE DESTINATÁRIO, aplicar KMODES, visualizar outlier superior VALOR UNITÁRIO

-> REMOVER NOME DA EMPRESA NO ARTIGO

5 - É possível identificar casos inconsistentes utilizando classificação de NCM?


date_variables = ['DATA EMISSÃO']
text_variables = ['NCM/SH (TIPO DE PRODUTO)','DESCRIÇÃO DO PRODUTO/SERVIÇO', 'NATUREZA DA OPERAÇÃO']
category_variables = ['CFOP', 'UNIDADE', 'CÓDIGO NCM/SH','PRESENÇA DO COMPRADOR', 'INDICADOR IE DESTINATÁRIO', 'UF EMITENTE', 'MUNICÍPIO EMITENTE', 'RAZÃO SOCIAL EMITENTE', 'UF DESTINATÁRIO', 'CONSUMIDOR FINAL', 'NOME DESTINATÁRIO','EVENTO MAIS RECENTE','DATA EMISSÃO MES', 'DATA EMISSÃO DIA SEMANA', 'DATA EMISSÃO PERIODO','DATA EMISSÃO DIA']
num_variables = ['VALOR NOTA FISCAL','QUANTIDADE','VALOR UNITÁRIO','VALOR TOTAL']