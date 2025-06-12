from flask import Flask
import os

def create_app():

    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    # Zip-Funktion zu Jinja2 Globalvariablen hinzufügen, damit Templates zip() direkt aufrufen können
    app.jinja_env.globals.update(zip=zip)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

    with app.app_context():
        from .api import api_blueprint
        app.register_blueprint(api_blueprint)
        return app