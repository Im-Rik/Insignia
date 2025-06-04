from flask import Blueprint, jsonify

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({"message": "Hello from Flask backend!"})
