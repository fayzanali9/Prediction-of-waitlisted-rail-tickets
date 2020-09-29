from flask import Flask, render_template, request
import ml

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/input', methods=["GET", "POST"])
def input():
    if request.method == 'GET':
        return render_template('input.html')
    elif request.method == 'POST':
        cocl = int(request.form['Coach class'])
        bks = int(request.form['Booking status'])
        sa1d = int(request.form['Status after 1 day'])
        sa1m = int(request.form['Status after 1 month'])
        sa1w = int(request.form['Status after 1 week'])
        sa2d = int(request.form['Status after 2 days'])
        algorithm = int(request.form['Algorithm'])


        algorithm_name = ''
        accuracy = 0
        if algorithm == 0:
            label = ml.classifier_gb.predict([[cocl,bks, sa1d,sa1m,sa2d,sa1w]])
            algorithm_name = 'GradientBoost'
            accuracy = ml.accuracy_gb

        elif algorithm == 1:
            label = ml.classifier_ada.predict([[cocl,bks, sa1d,sa1m ,sa2d,sa1w]])
            algorithm_name = 'Ada Boost'
            accuracy = ml.accuracy_ada
        elif algorithm == 2:
            label = ml.classifier_dt.predict([[cocl,bks, sa1d,sa1m ,sa2d,sa1w]])
            algorithm_name = 'Decision Tree'
            accuracy = ml.accuracy_dt
        elif algorithm == 3:
            label = ml.classifier_xg.predict([[cocl,bks, sa1d,sa1m ,sa2d,sa1w]])
            algorithm_name = 'XGBoost'
            accuracy = ml.accuracy_xg
        elif algorithm == 4:
            label = ml.classifier_rf.predict([[cocl, bks, sa1d,sa1m, sa2d, sa1w]])
            algorithm_name = 'Random Forest'
            accuracy = ml.accuracy_rf
        elif algorithm == 5:
            label = ml.classifier_svc.predict([[cocl, bks, sa1d,sa1m, sa2d, sa1w]])
            algorithm_name = 'SVM'
            accuracy = ml.accuracy_svc
        elif algorithm == 6:
            label = ml.classifier_nb.predict([[cocl, bks, sa1d,sa1m,sa2d, sa1w]])
            algorithm_name = 'Naive Bayes'
            accuracy = ml.accuracy_nb

        # print(f"purchased: {purchased[0]}")
        # if label == 0:
        #     label = "confirm"
        # else:
        #     label  = "not confirm"
        return render_template("result1.html", label=label, accuracy=accuracy, algorithm=algorithm_name)




app.run(port=4000, host='0.0.0.0', debug=True)