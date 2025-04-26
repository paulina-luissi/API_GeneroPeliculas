#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from p1_model_deployment import predict_popularity

# Initialize Flask app
app = Flask(__name__)
api = Api(
    app, 
    version='1.0', 
    title='API - Predicción Puntaje Popularidad Canciones',
    description='Predicts a popularity score from Spotify track features'
)

ns = api.namespace('predict', description='Prediction operations')

# Define the input arguments   
input_model = api.model('InputFeatures', {
    'track_id': fields.String(required=True, description='Unique ID of the track'),
    'artists': fields.String(required=True, description='Name of the artist'),
    'album_name': fields.String(required=True, description='Album name'),
    'track_name': fields.String(required=True, description='Track name'),
    'track_genre': fields.String(required=True, description='Genre of the track'),
    'key': fields.Integer(required=True, description='Estimated overall key of the track'),
    'mode': fields.Integer(required=True, description='Mode (major=1, minor=0)'),
    'time_signature': fields.Integer(required=True, description='Estimated time signature'),
    'explicit': fields.Boolean(required=True, description='Whether the track is explicit'),
    'duration_ms': fields.Integer(required=True, description='Duration of the track in milliseconds'),
    'danceability': fields.Float(required=True, description='Danceability score'),
    'energy': fields.Float(required=True, description='Energy score'),
    'loudness': fields.Float(required=True, description='Loudness in dB'),
    'speechiness': fields.Float(required=True, description='Speechiness score'),
    'acousticness': fields.Float(required=True, description='Acousticness score'),
    'instrumentalness': fields.Float(required=True, description='Instrumentalness score'),
    'liveness': fields.Float(required=True, description='Liveness score'),
    'valence': fields.Float(required=True, description='Musical positiveness'),
    'tempo': fields.Float(required=True, description='Tempo in BPM')
})

# Define the response model
response_model = api.model('Prediction', {
    'result': fields.Float,
})

@ns.route('/')
class PopularityApi(Resource):
    @api.expect(input_model)
    @api.marshal_with(response_model)
    def post(self):
        raw_features = api.payload
        result = predict_popularity(raw_features)
        return {"result": result}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
