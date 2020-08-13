## Project Title
Intent Recognition

## About 
It's a classifier modeled for the task of predicting which of intent corresponds best to a user requests.It uses Python and Flask. Intent Detection is one of the text classification task which find its application in any chatbot platform,in identification of an email as personal or business,computer-vision etc.
## Demo-Preview
![](g.gif)
## Data
The data is hosted on [Github](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines).It contains various user queries categorized into seven intents.
Here are the intents:

    SearchCreativeWork (e.g. Find me the I, Robot television show)
    GetWeather (e.g. Is it windy in Boston, MA right now?)
    BookRestaurant (e.g. I want to book a highly rated restaurant for me and my boyfriend tomorrow night)
    PlayMusic (e.g. Play the last track from Beyoncé off Spotify)
    AddToPlaylist (e.g. Add Diamonds to my roadtrip playlist)
    RateBook (e.g. Give 6 stars to Of Mice and Men)
    SearchScreeningEvent (e.g. Check the showtimes for Wonder Woman in Paris) 
## Preprocessing
Following preprocessing was performed on the text data:
- Stopwords Removal
- Noise Removal
- Accented Words Removal
- Lemmatization
## Modeling
### Machine Learning Method
- [Random Forests](https://github.com/Neha1529/Intent_Recognition/blob/master/model.py)
### Deep Learning Method
- [Bi-Directional LSTM](https://github.com/Neha1529/IntentRecognition/blob/master/Intent_recognition_using_LSTM.ipynb)
## How to Run 
1. Install Docker
2. Pull docker image from the [registry](https://hub.docker.com/repository/docker/neha2915/flask-app) by the command
    **docker pull neha2915/flask-app:intentimage**
3. Run the image by the command **docker run --name (name the container) -p 80:80 neha2915/flask-app:intentimage**
4. To view the output,on your browser go to **localhost:80** or **(ip-address of docker machine):80**
5. Type in a query in the text box eg. rate Hands on NLP 5 stars,click on predict to see the output
