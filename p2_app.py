#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from p2_model_deployment import predict_genre

# Initialize Flask app
app = Flask(__name__)
api = Api(
    app, 
    version='1.0', 
    title='API - Predicción de género de películas',
    description='Predicts the probability of the film of corresponding to multiple genres'
)

ns = api.namespace('predict', description='Prediction operations')

# Define the input arguments  
input_model = api.model('InputFeatures', {
    'title': fields.String(required=True, description='Film title'),
    'plot': fields.String(required=True, description='Film plot'),    
    'year': fields.Integer(required=True, description='Year of release of the film'),
    
})

# Define the response model
# response_model = api.model('Prediction', {
#     'result': fields.Float,
# })

genre_prob_model = api.model('GenreProbs', {
    'result': fields.Raw(description='Diccionario con géneros y probabilidades'),
})


@ns.route('/')
class GenrePrediction(Resource):
    @api.expect(input_model)
    @api.marshal_with(genre_prob_model)
    def post(self):
        raw_features = api.payload
        result = predict_genre(raw_features)
        return {"result": result}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
