from flask import Blueprint

predict_api_blueprint = Blueprint("predict_api_blueprint", __name__)

from . import routes