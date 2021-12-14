# spotify-song-classifier
Final project for Functional Programming in Software Engineering, Fall 2021, Johns Hopkins University.

Command-line application built in OCaml for classifying spotify songs into playlists, based on the audio features provided by the [Spotify Web API](https://developer.spotify.com/documentation/web-api/). Given any two playlists, the user can train and save a binary classifier, and then feed it a new song, to predict which of the two playlists the song "belongs" to. 

# Authors 
Jack Van Holland, Angi Benton

# Spotify Song and Playlist IDs 
Spotify provides unique identifiers for objects (song, album, artist, playlist, etc.). This project requires the user to input songs and playlists by their spotify ID because names can be ambiguous. The IDs are most easily accessed from the URL of the song/playlist, found through "share" -> "copy link". The ID is the last part of the path, before the question mark. 

For example, here is the URL of a song:

https://open.spotify.com/track/5CQ30WqJwcep0pYcV4AMNc?si=05dd7edaf8c54367

And here is the ID:

5CQ30WqJwcep0pYcV4AMNc

# Python Virtual Environment 
This command line application needs to be run within a python virtual environment due to its dependency on python libraries. 


### Create and activate (first use):

```
$ python3 -mvenv .venv 
$ source .venv/bin/activate 
$ pip install scikit-learn==0.23.2 numpy==1.19.4 scipy==1.5.4 pytest 
$ [ run your dune commands - see Usage ] 
$ deactivate  
```

### Activate (subsequent uses): 

```
$ source .venv/bin/activate 
$ [ run your dune commands - see Usage] 
$ deactivate  
```

# Usage 
The following example is also provided as a bash script, ./testworkflow.sh, with explanatory comments. 

To compile the project:
```
$ dune build 
```

Then you can run the single executable, src/run.exe, in 4 different modes:
```
$ dune exec -- ./src/run.exe [MODE] [OPTIONS]
```

## Mode: download 
Fetches the song feature data about the first 100 songs of two given playlists, transforms the data into model-friendly format, and saves it to a subfolder inside ./datasets. 

Example: 
```
$ dune exec -- ./src/run.exe download --pos-id 1h0CEZCm6IbFTbxThn6Xcs --neg-id 68PbMiU12ssvm5UAHsqqOi --dataset-folder trap_vs_classical --test 0.2 --val 0.1
```
Required:
* **--pos-id** = Spotify ID for the playlist to be assigned +1 as a label 
* **--neg-id** = Spotify ID for the playlist to be assigned -1 as a label 
* **--dataset-folder** = a name for the folder that will be created to hold the cleaned dataset 
* **--test** = fraction of the data to reserve for testing 
* **--test** = fraction of the data to reserve for validation
  
Optional: 
* **--standardize** = center each feature around 0 with unit standard deviation
* **--normalize** = normalize each feature in the range (0,1)
* **--balance** = subsample from classes to equalize quanities, by default included
* **--random** = randomize order of samples


## Mode: train
Train a binary classifier (SVM) on a dataset and save the model. 

Example: 
```
$ dune exec -- ./src/run.exe train --dataset-folder  trap_vs_classical --model-file trap_vs_classical_model.txt
```
Required:
* **--dataset-folder** = folder you gave during 'download' mode
* **--model-file** = new file to save the model in
  
Optional: 
* **--c**  = Specify the regularization strength. Defaults to 1.0, 
                 or if multiple provided, must provide evaluation metric
                 and will automatically tune to the best model.
* **--metric**  = Metric to optimize in tuning (either "accuracy" or "f1")

## Mode: test
Test a binary classifier (SVM) on the test portion of a dataset and print the results. 

Example: 
```
$ dune exec -- ./src/run.exe test --dataset-folder  trap_vs_classical --model-file trap_vs_classical_model.txt
```
Required:
* **--dataset-folder** = folder you gave during 'download' mode
* **--model-file** = filename you gave during 'train' mode



## Mode: classify 
Fetch the feature information for a song id and use a pretrained model to classify it.  

Example: 
```
$ dune exec -- ./src/run.exe classify --model-file trap_vs_classical_model.txt --song-id 1LNqr2FALBNFX1smNJcKD2
```
Required:
* **--model-file**  filename you gave during 'train' mode 
* **--song-id**  Spotify ID of a new song (may or may not be a member of either playlists)

# Testing 
To test our libraries, run
```
$ dune test
```


# Dependencies 
* [Spotify Web API](https://developer.spotify.com/documentation/web-api/) 
* [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
* [ounit2](https://github.com/gildor478/ounit)
* [lwt](https://github.com/ocsigen/lwt) 
* [cohttp-lwt-unix](https://github.com/mirage/ocaml-cohttp)
* [yojson](https://github.com/ocaml-community/yojson) 
* [ocaml sklearn](https://github.com/lehy/ocaml-sklearn) 
* [ocaml numpy](https://github.com/LaurentMazare/npy-ocaml)

