open Core
open Svm
open Spotify
open Machine_learning

let download = 
  Command.basic
    ~summary:"Download and save a dataset from two Spotify playlists"
    Command.Let_syntax.(
      let%map_open pos_id = flag "--pos-id" (required string)
          ~doc:" The positive playlist id"
      and neg_id = flag "--neg-id" (required string)
          ~doc:" The negative playlist id"
      and filename = flag "--dataset-folder" (required Filename.arg_type)
          ~doc:" The file location to store the model"
      and norm = flag "--normalize" (no_arg)
          ~doc:"Option to normalize each feature in the range (0,1)"
      and standard = flag "--standardize" (no_arg)
          ~doc:"Option to center each feature around 0 
        with unit standard deviation"
      and balance = flag "--balance" (optional_with_default true bool)
          ~doc:"Option to subsample from classes to equalize quanities, 
          by default included"
      and random = flag "--random" (optional_with_default true bool)
          ~doc:"Option to randomize order of samples, 
          by default random but can be specified for deterministic testing"
      and valid = flag "--val" (required float)
          ~doc:" The proportion (0,1) of the dataset to allocate to validation"
      and test = flag "--test" (required float)
          ~doc:" The proportion (0,1) of the dataset to allocate to testing"
      in 
      fun () -> if norm && standard 
        then failwith "Can only choose one out of --standardize and --normalize" 
        else let download_monadic = Stdio.print_string "Loading playlist 1…\n"; 
               let%lwt token = get_new_api_token () in 
               let%lwt pos = playlist_of_id pos_id token in
               Stdio.print_string "Loading playlist 2…\n";
               let%lwt neg = playlist_of_id neg_id token in
               Stdio.print_string "Processing playlists into dataset\n"; 
               let save ((pos, neg): (playlist * playlist)) : unit = 
                 (if random then (SVM_Classification.randomize pos, 
                                  SVM_Classification.randomize neg) else (pos, neg))
                 |> fun (pos, neg) -> (if norm 
                                       then SVM_Classification.normalize (pos, neg) 
                                       else if standard 
                                       then SVM_Classification.standardize (pos, neg)
                                       else (pos, neg, [])) |> fun (pos, neg, preprocess) ->  
                                      (if balance 
                                       then SVM_Classification.balance_classes 
                                           (pos, neg) else (pos, neg))
                                      |> fun classes 
                                      -> SVM_Classification.split classes valid test preprocess 
                                         |> fun d 
                                         -> Stdio.print_string 
                                         @@ "Saving dataset to ./datasets/...\n" 
                                            ^ filename; 
                                         SVM_Classification.save_dataset d 
                                         @@ "./datasets/" ^ filename;
               in (pos, neg) |> save |> fun () -> Stdio.print_string "Saved dataset.\n"; 
                  Lwt.return ();
          in Lwt_main.run download_monadic)

let train = 
  Command.basic
    ~summary:"Train a machine learning model based on two playlists"
    Command.Let_syntax.(
      let%map_open dataset_folder = flag "--dataset-folder" (required Filename.arg_type)
          ~doc:" The folder name of the dataset (inside ./datasets/)"
      and model_file = flag "--model-file" (required Filename.arg_type)
          ~doc:" The file name to store the model (inside ./models/)"
      and cs = flag "--c" (listed float)
          ~doc: "Specify the regularization strength. Defaults to 1.0, 
                 or if multiple provided, must provide evaluation metric
                 and will automatically tune to the best model."
      and metric = flag "--metric" (optional_with_default "" string)
          ~doc: "Metric to optimize in tuning (either \"accuracy\" or \"f1\")"
      in
      fun () -> if 0 = String.length metric && 1 < List.length cs 
        then failwith "Must provide evalutation metric optimizer if multiple hyperparameters
           provided for tuning" else
          Stdio.print_string "Loading dataset"; 
        SVM_Classification.load_dataset @@ "./datasets/" ^ dataset_folder |>
        fun d -> Stdio.print_string "Training model…\n"; 
        (match cs with
         | [] -> SVM_Model.train {reg = 1.0; shift = d.shift; scale = d.scale} d.pos_train d.neg_train
         | c :: [] -> SVM_Model.train {reg = c; shift = d.shift; scale = d.scale}  d.pos_train d.neg_train 
         | _ -> SVM_Model.tune (List.map cs ~f:(fun c : SVM_Model.hyperparameters -> {reg = c; shift = d.shift; scale = d.scale})) d.pos_train d.neg_train 
                |> fun svms -> SVM_Classification.tune svms d.pos_valid d.neg_valid 
                  (if String.equal "accuracy" metric 
                   then SVM_Classification.accuracy else SVM_Classification.f1_score)) 
        |> fun (svm) -> Stdio.print_string @@ "Saving model to " ^ model_file ^ "...\n"; 
        SVM_Model.save svm @@ "./models/" ^ model_file
        |> fun () -> Stdio.print_string "Saved model.\n")

let classify =
  Command.basic
    ~summary:"Classify a song based on an existing model"
    Command.Let_syntax.(
      let%map_open filename = flag "--model-file" (required Filename.arg_type)
          ~doc:" The file location to retrieve the model"
      and song = flag "--song-id" (required string)
          ~doc:" The song id"
      in
      fun () -> let classify_monadic = Stdio.print_string 
                  @@ "Loading model from " ^ "./models/" ^ filename ^ "...\n";
                  SVM_Model.load @@ "./models/" ^ filename 
                  |> fun (svm) -> let%lwt token = get_new_api_token () in 
                  Stdio.print_string "Loading song..."; 
                  let%lwt s = song_of_id song token 
                  in Stdio.print_string "Classifying song...\n"; 
                  SVM_Classification.classify svm s
                  |> fun (p) -> Stdio.print_string 
                  @@ "Song classified as a member of \"" ^ p ^ "\""; 
                  Lwt.return ();
        in Lwt_main.run classify_monadic)

let test =
  Command.basic
    ~summary:"Test a collection of songs with true labels"
    Command.Let_syntax.(
      let%map_open model_file = flag "--model-file" (required Filename.arg_type)
          ~doc:" The file location to retrieve the model"
      and dataset_folder = flag "--dataset-folder" (required Filename.arg_type)
          ~doc:" The positive test playlist id"
      in
      fun () -> Stdio.print_string @@ "Loading dataset from " ^ "./datasets/" ^ dataset_folder; 
        SVM_Classification.load_dataset @@ "./datasets/" ^ dataset_folder |>
        fun d -> 
        Stdio.print_string @@ "Loading model from " ^ "./models/" ^ model_file ^ "...\n";
        SVM_Model.load @@ "./models/" ^ model_file |> fun (svm) -> Stdio.print_string 
        @@ "Classifying test examples...\n";
        SVM_Classification.test svm d.pos_test d.neg_test 
        |> fun(cm) -> Stdio.print_string @@ "Confusion matrix: \n" 
                                            ^ SVM_Classification.pretty_confusion cm 
                      |> fun () -> Stdio.printf "Accuracy is %f.\n" 
                      @@ SVM_Classification.accuracy cm 
                                   |> fun () -> Stdio.printf "F1 score is %f.\n" 
                                   @@ SVM_Classification.f1_score cm;)

let command =
  Command.group ~summary:"Perform machine learning on any public playlists
   and songs with the Spotify API"
    [ "download", download; "train", train
    ; "classify", classify ; "test", test]

let () = Command.run command