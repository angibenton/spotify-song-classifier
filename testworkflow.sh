#!/bin/bash

# Example Usage for Spotify Song Classifier 

# test playlists contain classical and trap music
CLASSICALPLAYLISTID=1h0CEZCm6IbFTbxThn6Xcs
TRAPPLAYLISTID=68PbMiU12ssvm5UAHsqqOi

# test song is Moonlight Sonata, expected to be classified as classical 
MOONLIGHTSONGID=1LNqr2FALBNFX1smNJcKD2

# test playlists contain classical and trap music
DATASETFOLDER=trap_vs_classical
MODELFILE=trap_vs_classical_model.txt

# these will exist if the script was already ran, and need to be deleted 
rm -r ./datasets/$DATASETFOLDER 
rm ./models/$MODELFILE

# ENTER VENV (see README.md for venv setup)
source .venv/bin/activate

# build the project  
dune build 

# execute run.exe with different options to fetch the playlists, train the model, test it, fetch the song, and classify it
dune exec -- ./src/run.exe download --pos-id $TRAPPLAYLISTID --neg-id $CLASSICALPLAYLISTID --dataset-folder $DATASETFOLDER --test 0.2 --val 0.1 --standardize
dune exec -- ./src/run.exe train --dataset-folder $DATASETFOLDER --model-file $MODELFILE  
dune exec -- ./src/run.exe test --dataset-folder $DATASETFOLDER --model-file $MODELFILE
dune exec -- ./src/run.exe classify --model-file $MODELFILE --song-id $MOONLIGHTSONGID

# EXIT VENV 
deactivate 

