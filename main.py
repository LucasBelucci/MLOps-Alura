from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os

'''
dados = pd.read_csv(
    'https://caelum-online-public.s3.amazonaws.com/1576-mlops/casas.csv')

colunas = ['tamanho', 'ano', 'garagem']


x = dados.drop('preco', axis=1)
y = dados['preco']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(x_train, y_train)
'''

modelo = pickle.load(open('models\modelo.sav', 'rb'))
colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
# 'lucas'
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
# 'senha'

basic_auth = BasicAuth(app)


@app.route('/')
def home():
    return 'Minha primeira API.'


@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity
    return 'Polaridade: {}'.format(polaridade)


@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
