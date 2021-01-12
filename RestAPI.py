import json
import Main

from functools      import wraps
from flask_restful  import Resource, Api
from flask          import Flask, request, jsonify, abort

app = Flask(__name__)
api = Api(app)

# The actual decorator function
def require_appkey(view_function):
	@wraps(view_function)
	# the new, post-decoration function. Note *args and **kwargs here.
	def decorated_function(*args, **kwargs):
		with open('api.key', 'r') as apikey:
			key = apikey.read().replace('\n', '')
		if request.headers.get('x-api-key') and request.headers.get('x-api-key') == key:
			return view_function(*args, **kwargs)
		else:
			abort(401)
	return decorated_function

class Emotion(Resource):
	@require_appkey
	def post(self):
		text = request.json['text']
		clear_text = Main.text_preprocessing(text)
		return clear_text

# Routes
api.add_resource(Emotion, '/api/v1/preprocessing')  

# Main
if __name__ == '__main__':
	app.run(port='8390')