from flask import Flask, send_file
from main import run_trading_bot
import os

app = Flask(__name__)

@app.route("/")
def home():
    run_trading_bot()
    result_path = "enhanced_signals_with_scores.csv"
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
