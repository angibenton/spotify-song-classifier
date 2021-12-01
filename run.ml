open Core
open Svm

let playlist_of_id = fun (s) ->
  match s with
  | "1" ->{name = "the first one"; id = "123"; 
           features = (Np.matrixf [| [| -1.; -1. |]; [| -2.; -1. |]; [| 4.; 3.2|]; [| 2.; 1. |] |])}
  | _ -> {name = "the second one"; id = "124"; features = (Np.matrixf [| [| -1.; -1. |];  |])}

let train = 
  Command.basic
    ~summary:"Train a machine learning model based on two playlists"
    Command.Let_syntax.(
      let%map_open playlist1 = flag "--playlist-id-1" (required string)
          ~doc:"The first playlist id"
      and playlist2 = flag "--playlist-id-2" (required string)
          ~doc:"The second playlist id"
      and filename = flag "--model-file" (required Filename.arg_type)
          ~doc:"The first playlist id"
      and acc = flag "--accuracy" (no_arg)
          ~doc:"Compute and display the training accuracy"
      and f1 = flag "--f1" (no_arg)
          ~doc:"Compute and display the training F1 score"
      in
      fun () -> Stdio.print_string "Loading playlist 1…\n"; 
        playlist_of_id playlist1 
        |> fun p1 -> Stdio.print_string "Loading playlist 2…\n"; 
        (p1, playlist_of_id playlist2) 
        |> fun (p1, p2) -> Stdio.print_string "Training model…\n"; 
        SVM_Model.train p1 p2 
        |> fun (svm) -> if acc || f1 then Stdio.print_string "Evaluating model on training set...\n"; 
        SVM_Classification.test svm p1 p2 
        |> fun (cm) -> (if acc then Stdio.printf "Training accuracy is %f.\n" 
                          @@ SVM_Classification.accuracy cm else ()) 
                       |> fun () -> (if f1 then Stdio.printf "Training F1 score is %f.\n" @@ SVM_Classification.f1_score cm 
                                     else ()) |> fun () -> Stdio.print_string @@ "Saving model to " ^ filename ^ "...\n"; 
                                    SVM_Model.save svm filename
                                    |> fun () -> Stdio.print_string "Saved model.\n")
let classify =
  Command.basic
    ~summary:"Classify a song based on an existing model"
    Command.Let_syntax.(
      let%map_open base = anon ("base" %: date)
      and days = anon ("days" %: int)
      in
      fun () ->
        Date.add_days base days
        |> Date.to_string
        |> print_endline)
let test =
  Command.basic
    ~summary:"Test a collection of songs with true labels"
    Command.Let_syntax.(
      let%map_open base = anon ("base" %: date)
      and days = anon ("days" %: int)
      in
      fun () ->
        Date.add_days base days
        |> Date.to_string
        |> print_endline)

let command =
  Command.group ~summary:"Perform machine learning on some playlists and songs with the Spotify API"
    [ "train", train
    ; "classify", classify ; "test", test]

let () = Command.run command