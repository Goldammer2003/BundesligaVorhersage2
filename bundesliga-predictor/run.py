from app import create_app

app = create_app()

@app.route('/debug')
def debug():
    return str(app.static_folder)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
