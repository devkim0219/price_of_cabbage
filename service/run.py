from flask import Flask, render_template, request, redirect, jsonify
import pickle

app = Flask(__name__)

# 모델 불러오기
with open('./model/predict_cabbage_price.model', 'rb') as f:
    sc = pickle.load(f)
    mlr = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        # ajax 통신을 통한 post 방식의 파라미터 받아오기
        avg_temp = request.form['avg_temp']
        min_temp = request.form['min_temp']
        max_temp = request.form['max_temp']
        rainfall = request.form['rainfall']

        # 위에서 받아온 값을 모델에 적용하기 위해 array 형태로 바꿔줌 
        X = [[avg_temp, min_temp, max_temp, rainfall]]
        print(X)

        # 모델에 정규화 된 데이터를 넣어 예측값 출력
        X_std = sc.transform(X)
        print(X_std)

        predicted_price = mlr.predict(X_std)
        print(predicted_price)

        # 예측값 소수점 자리 반올림
        price = round(predicted_price[0][0])

        return jsonify({'price': price})

if __name__ == '__main__':
    app.run(debug=True)