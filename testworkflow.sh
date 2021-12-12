#!/bin/bash

CLASSICALPLAYLISTID=1h0CEZCm6IbFTbxThn6Xcs
TRAPPLAYLISTID=68PbMiU12ssvm5UAHsqqOi
MOONLIGHTSONGID=1LNqr2FALBNFX1smNJcKD2


DATASETFOLDER=trap_vs_classical
MODELFILE=trap_vs_classical_model.txt

source .venv/bin/activate
dune build 
rm -r ./datasets/$DATASETFOLDER
dune exec -- ./src/run.exe download --pos-id $TRAPPLAYLISTID --neg-id $CLASSICALPLAYLISTID --dataset-folder $DATASETFOLDER --test 0.2 --val 0.2 --standardize
dune exec -- ./src/run.exe train --dataset-folder $DATASETFOLDER --model-file $MODELFILE  
dune exec -- ./src/run.exe test --dataset-folder $DATASETFOLDER --model-file $MODELFILE
dune exec -- ./src/run.exe classify --model-file $MODELFILE --song-id $MOONLIGHTSONGID
deactivate 

