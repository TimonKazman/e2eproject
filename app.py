from flask import Flask, request, render_template
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # date_string = request.form['date']
        # date_range = pd.date_range(date_string, periods=24, freq='H')
        # df = pd.DataFrame(date_range, columns=['Date'])
        # df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        date_string = CustomData(
            request.form['date']
        )

        df = date_string.get_data_as_data_frame()
        print(df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(df)
        
        return render_template("home.html", results = results[0])
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug=True)