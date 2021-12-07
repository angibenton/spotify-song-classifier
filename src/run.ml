open Core
open Svm
open Spotify

let train = 
  Command.basic
    ~summary:"Train a machine learning model based on two playlists"
    Command.Let_syntax.(
      let%map_open pos_id = flag "--pos-id" (required string)
          ~doc:" The positive playlist id"
      and neg_id = flag "--neg-id" (required string)
          ~doc:" The negative playlist id"
      and filename = flag "--model-file" (required Filename.arg_type)
          ~doc:" The file location to store the model"
      and c = flag "--c" (optional_with_default 1.0 float)
          ~doc:" Specify the regularization strength"
      in
      fun () -> let train_monadic = Stdio.print_string "Loading playlist 1…\n"; 
                  let%lwt token = get_new_api_token () in 
                  let%lwt pos = playlist_of_id pos_id token in
                  Stdio.print_string "Loading playlist 2…\n";
                  let%lwt neg = playlist_of_id neg_id token in
                  Stdio.print_string "Training model…\n"; 
                  SVM_Model.train c pos neg 
                  |> fun (svm) -> Stdio.print_string @@ "Saving model to " ^ filename ^ "...\n"; 
                  SVM_Model.save svm filename
                  |> fun () -> Stdio.print_string "Saved model.\n"; Lwt.return ();
        in Lwt_main.run train_monadic)
let classify =
  Command.basic
    ~summary:"Classify a song based on an existing model"
    Command.Let_syntax.(
      let%map_open filename = flag "--model-file" (required Filename.arg_type)
          ~doc:" The file location to retrieve the model"
      and song = flag "--song-id" (required string)
          ~doc:" The song id"
      in
      fun () -> let classify_monadic = Stdio.print_string @@ "Loading model from " ^ filename ^ "...\n";
                  SVM_Model.load filename |> fun (svm) -> let%lwt token = get_new_api_token () in 
                  Stdio.print_string "Loading song..."; let%lwt s = song_of_id song token in  Stdio.print_string "Classifying song...\n"; 
                  SVM_Classification.classify svm s
                  |> fun (p) -> Stdio.print_string @@ "Song classified as a member of \"" ^ p ^ "\""; Lwt.return ();
        in Lwt_main.run classify_monadic)

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
      fun () -> let test_monadic = 
                  let%lwt token = get_new_api_token () in 
                  let%lwt pos = playlist_of_id pos_id token in
                  Stdio.print_string "Loading playlist 2…\n";
                  let%lwt neg = playlist_of_id neg_id token in
                  Stdio.print_string "Training model…\n";  Stdio.print_string @@ "Loading model from " ^ model ^ "...\n";
                  SVM_Model.load model |> fun (svm) -> Stdio.print_string @@ "Classifying test examples...\n";
                  SVM_Classification.test svm pos neg |> fun(cm) -> Stdio.print_string 
                  @@ "Confusion matrix: \n" ^ SVM_Classification.pretty_confusion cm 
                                                                    |> fun () -> Stdio.printf "Accuracy is %f.\n" 
                                                                    @@ SVM_Classification.accuracy cm 
                                                                                 |> fun () -> Stdio.printf "F1 score is %f.\n" 
                                                                                 @@ SVM_Classification.f1_score cm; Lwt.return ();
        in Lwt_main.run test_monadic

    )

let command =
  Command.group ~summary:"Perform machine learning on any public playlists and songs with the Spotify API"
    [ "train", train
    ; "classify", classify ; "test", test]

let () = Command.run command