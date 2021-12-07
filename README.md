# spotify-song-classifier
Final project for Functional Programming in Software Engineering, Fall 2021, Johns Hopkins University 

## Authors 
Jack Van Holland, Angi Benton

## Python Virtual Environment 
This command line application needs to be run within a python virtual environment due to its dependency on python libraries. 

### Create and activate (first use):

$ python3 -mvenv .venv \
$ source .venv/bin/activate \
$ pip install scikit-learn==0.23.2 numpy==1.19.4 scipy==1.5.4 pytest \
$ [use spotify-song-classifier as desired] \
$ deactivate  

### Activate (subsequent uses): 

$ source .venv/bin/activate \
$ [use spotify-song-classifier as desired] \
$ deactivate  

## Testing
To unit test most of the functionality, run \
$ dune test \ 
To test the spotify API functionality, run \
$ dune build \ 
$ dune exec ./src/test_spotify.exe 
