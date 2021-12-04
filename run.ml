open Core
open Machine_learning
open Svm

let playlist_of_id = fun (s) ->
  match s with
  | "1" ->{name = "the first one"; pid = "123"; 
           features_matrix = (Np.matrixf [| [| -1.; -1. |]; [| -2.; -1. |]; [| 4.; 3.2|]; [| 2.; 1. |] |])}
  | _ -> {name = "the second one"; pid = "124"; features_matrix = (Np.matrixf [| [| -5.; -5. |];  |])}

let song_of_id _ : song = {features_vector = Np.vectorf [| -5.; -5. |]; name = "hi"; sid = "h"}

let train = 
  Command.basic
    ~summary:"Train a machine learning model based on two playlists"
    Command.Let_syntax.(
      let%map_open playlist1 = flag "--playlist-id-1" (required string)
          ~doc:" The first playlist id"
      and playlist2 = flag "--playlist-id-2" (required string)
          ~doc:" The second playlist id"
      and filename = flag "--model-file" (required Filename.arg_type)
          ~doc:" The file location to store the model"
      and c = flag "--c" (optional_with_default 1.0 float)
          ~doc:" Specify the regularization strength"
      in
      fun () -> Stdio.print_string "Loading playlist 1…\n"; 
        playlist_of_id playlist1 
        |> fun p1 -> Stdio.print_string "Loading playlist 2…\n"; 
        (p1, playlist_of_id playlist2) 
        |> fun (p1, p2) -> Stdio.print_string "Training model…\n"; 
        SVM_Model.train c p1 p2 
        |> fun (svm) -> Stdio.print_string @@ "Saving model to " ^ filename ^ "...\n"; 
                                    SVM_Model.save svm filename
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
      fun () -> Stdio.print_string @@ "Loading model from " ^ filename ^ "...\n";
        SVM_Model.load filename |> fun (svm) -> Stdio.print_string "Classifying song...\n";
        SVM_Classification.classify svm @@ song_of_id song 
        |> fun (p) -> Stdio.print_string @@ "Song classified as a member of \"" ^ p ^ "\"")

let test =
  Command.basic
    ~summary:"Test a collection of songs with true labels"
    Command.Let_syntax.(
      let%map_open model = flag "--model-file" (required Filename.arg_type)
          ~doc:" The file location to retrieve the model"
      and pos_id = flag "--pos-id" (required string)
      ~doc:" The positive test playlist id"
      and neg_id = flag "--neg-id" (required string)
      ~doc:" The negative test playlist id"
      in
      fun () -> 
        Stdio.print_string "Loading positive test playlist…\n"; 
        playlist_of_id pos_id 
        |> fun pos -> Stdio.print_string "Loading negative test playlist…\n"; 
        (pos, playlist_of_id neg_id) 
        |> fun (pos, neg) -> Stdio.print_string @@ "Loading model from " ^ model ^ "...\n";
      SVM_Model.load model |> fun (svm) -> Stdio.print_string @@ "Classifying test examples...\n";  SVM_Classification.test svm pos neg
        |> fun(cm) -> Stdio.print_string @@ "Confusion matrix: \n" ^ SVM_Classification.pretty_confusion cm |> fun () -> Stdio.printf "Accuracy is %f.\n" @@ SVM_Classification.accuracy cm |> fun () -> Stdio.printf "F1 score is %f.\n" @@ SVM_Classification.f1_score cm 
    )

let command =
  Command.group ~summary:"Perform machine learning on any public playlists and songs with the Spotify API"
    [ "train", train
    ; "classify", classify ; "test", test]

let () = Command.run command