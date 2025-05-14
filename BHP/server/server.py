from flask import Flask, request, jsonify
import util
import webbrowser
import os
app = Flask(__name__)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin','*')

    return response


@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
    })

    response.headers.add('Access-Control-Allow-Origin','*')

    return response

def open_browser():
    # Get the absolute path to the client directory
    client_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'client', 'app.html'))
    # Convert to file URL format
    file_url = 'file:///' + client_path.replace('\\', '/')
    webbrowser.open(file_url)

if __name__=="__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    # Open browser after a short delay to ensure server is running
    import threading
    threading.Timer(1.5, open_browser).start()
    app.run()