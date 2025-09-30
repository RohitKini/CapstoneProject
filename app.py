from flask import Flask, render_template, request
from model import get_recommendations_for_user

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    username = None
    recommendations = None

    if request.method == 'POST':
        username = request.form['username'].strip()
        print(f"Fetching recommendations for user: {username}")
        recommendations = get_recommendations_for_user(username)

    return render_template(
        'index.html',
        username=username,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
