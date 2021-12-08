open Core
open OUnit2
open Machine_learning
open Svm
open Spotify

let float_equals_epsilon = fun f1 f2 -> Float.(-) f1 f2 |> Float.abs |>  Float.(>) 0.001;;

let cm_1 = {tp = 5; fp = 0; fn = 6; tn = 4};;
let cm_2 = {tp = 55431; fp = 5; fn = 58520; tn = 314};;
let cm_3 = {tp = 9; fp = 588097; fn = 907; tn = 78954};;
let pretty_confusion_1 = "               actual  
               pos neg 
              ---------
          pos | 5 | 0 |
predicted     ---------
          neg | 6 | 4 |
              ---------
";;
let pretty_confusion_2 = "                   actual      
                  pos   neg    
              -----------------
          pos | 55431 |     5 |
predicted     -----------------
          neg | 58520 |   314 |
              -----------------
";;
let pretty_confusion_3 = "                    actual       
                  pos     neg    
              -------------------
          pos |      9 | 588097 |
predicted     -------------------
          neg |    907 |  78954 |
              -------------------
";;

let test_pretty_confusion _ = 
  assert_equal (SVM_Classification.pretty_confusion cm_1) pretty_confusion_1;
  assert_equal (SVM_Classification.pretty_confusion cm_2) pretty_confusion_2;
  assert_equal (SVM_Classification.pretty_confusion cm_3) pretty_confusion_3;
;;

let test_acc _ =
  assert_bool "Confusion Matrix 1" (float_equals_epsilon 0.6 
                                    @@ SVM_Classification.accuracy cm_1);
  assert_bool "Confusion Matrix 2" (float_equals_epsilon 0.4878 
                                    @@ SVM_Classification.accuracy cm_2);
  assert_bool "Confusion Matrix 3" (float_equals_epsilon 0.1182 
                                    @@ SVM_Classification.accuracy cm_3);
;;

let test_f1 _ =
  assert_bool "Confusion Matrix 1" (float_equals_epsilon 0.6250 
                                    @@ SVM_Classification.f1_score cm_1);
  assert_bool "Confusion Matrix 2" (float_equals_epsilon 0.6545 
                                    @@ SVM_Classification.f1_score cm_2);
  assert_bool "Confusion Matrix 3" (float_equals_epsilon 0.
                                    @@ SVM_Classification.f1_score cm_3);
;;

let pos_song_1 = {name = "Positive Song 1"; sid = "1";
                  features_vector = Np.reshape ~newshape:[1; 1] 
                    @@ Np.Ndarray.vectorf [|1.|]}

let neg_song_1 = {name = "Negative Song 1"; sid = "1";
                  features_vector = Np.reshape ~newshape:[1; 1] 
                    @@ Np.Ndarray.vectorf [|0.5|]}

let pos_song_2 = {name = "Positive Song 2"; sid = "2";
                  features_vector = Np.reshape ~newshape:[1; 2] 
                    @@ Np.Ndarray.vectorf [| 10.; 10. |]}

let neg_song_2 = {name = "Negative Song 2"; sid = "2";
                  features_vector = Np.reshape ~newshape:[1; 2] @@  Np.Ndarray.vectorf [| -10.; -10. |]}

let pos_song_3 = {name = "Positive Song 1"; sid = "3";
                  features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                      [| 0.; 1.; 10.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 3.0; 0.5|]}

let neg_song_3 = {name = "Negative Song 3"; sid = "3";
                  features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                      [| 0.; 5.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; -3.0; 0.75|]}

let pos_playlist_1 = {name = "Positive Playlist 1"; pid = "123"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array pos_song_1.features_vector |])};;
let neg_playlist_1 = {name = "Negative Playlist 1"; pid = "124"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array neg_song_1.features_vector |])};;


let pos_playlist_2 = {name = "Positive Playlist 2"; pid = "123"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array pos_song_2.features_vector |])};;
let neg_playlist_2 = {name = "Negative Playlist 2"; pid = "124"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array neg_song_2.features_vector |])};;

let pos_playlist_3 = {name = "Positive Playlist 3"; pid = "123"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array pos_song_3.features_vector |])};;
let neg_playlist_3 = {name = "Negative Playlist 3"; pid = "124"; 
                      features_matrix = (Np.matrixf [|Np.Ndarray.to_float_array neg_song_3.features_vector |])};;

let svm_1 = SVM_Model.train 1.0 pos_playlist_1 neg_playlist_1
let svm_2 = SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2
let svm_3 = SVM_Model.train 1.0 pos_playlist_3 neg_playlist_3

let test_classify _ = 
  assert_equal "Positive Playlist 1" @@ SVM_Classification.classify svm_1 pos_song_1;
  assert_equal "Negative Playlist 1" @@ SVM_Classification.classify svm_1 neg_song_1;
  assert_equal "Positive Playlist 2" @@ SVM_Classification.classify svm_2 pos_song_2;
  assert_equal "Negative Playlist 2" @@ SVM_Classification.classify svm_2 neg_song_2;
  assert_equal "Positive Playlist 3" @@ SVM_Classification.classify svm_3 pos_song_3;
  assert_equal "Negative Playlist 3" @@ SVM_Classification.classify svm_3 neg_song_3;
;;

let true_cm = {tp = 1; fp = 0; fn = 0; tn = 1};;

let false_cm = {tp = 0; fp = 1; fn = 1; tn = 0};;


let confusion_equal cm_1 cm_2 : bool = 
  Int.equal cm_1.tp cm_2.tp && Int.equal cm_1.tn cm_2.tn 
  && Int.equal cm_1.fn cm_2.fn && Int.equal cm_1.fp cm_2.fp

let test_test _ = 
  SVM_Classification.test svm_1 pos_playlist_1 neg_playlist_1 
  |> fun cm -> assert_bool "Incorrect classification during test" @@ confusion_equal cm true_cm;
  SVM_Classification.test svm_1 neg_playlist_1 pos_playlist_1 
  |> fun cm -> assert_bool "Incorrect classification during test" @@ confusion_equal cm false_cm;
  SVM_Classification.test svm_2 pos_playlist_2 neg_playlist_2 
  |> fun cm -> assert_bool "Incorrect classification during test" @@ confusion_equal cm true_cm;
  SVM_Classification.test svm_2 neg_playlist_2 pos_playlist_2 
  |> fun cm -> assert_bool "Incorrect classification during test" @@ confusion_equal cm false_cm;
  SVM_Classification.test svm_3 pos_playlist_3 neg_playlist_3 
  |> fun cm -> assert_bool "Incorrect classification during test" @@ confusion_equal cm true_cm;
  SVM_Classification.test svm_3 neg_playlist_3 pos_playlist_3 
  |> fun cm -> assert_bool "Incorrect classification during test" @@ confusion_equal cm false_cm;

;;

let test_svm_train_predict _ = 
  assert_bool "Model 1 misclassifies its positive song" 
  @@ SVM_Model.predict svm_1 @@ pos_song_1.features_vector;
  assert_bool "Model 1 misclassifies its negative song" 
  @@ not @@ SVM_Model.predict svm_1 @@ neg_song_1.features_vector;
  assert_bool "SVM should train deterministically" (
    (SVM_Model.train 1.0 pos_playlist_1 neg_playlist_1, SVM_Model.train 1.0 pos_playlist_1 neg_playlist_1)
    |> fun (svm_1, svm_2) -> SVM_Model.equal svm_1 svm_2);
  assert_bool "Model 2 misclassifies its positive song" 
  @@ SVM_Model.predict svm_2 @@ pos_song_2.features_vector;
  assert_bool "Model 2 misclassifies its negative song" 
  @@ not @@ SVM_Model.predict svm_2 @@ neg_song_2.features_vector );
assert_bool "SVM should train deterministically" (
  (SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2, SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2)
  |> fun (svm_1, svm_2) -> SVM_Model.equal svm_1 svm_2);
assert_bool "Model 3 misclassifies its positive song" 
@@ SVM_Model.predict svm_3 @@ pos_song_3.features_vector);
assert_bool "Model 3 misclassifies its negative song" 
@@ not @@ SVM_Model.predict svm_3 @@ neg_song_3.features_vector);
assert_bool "SVM should train deterministically" (
  (SVM_Model.train 1.0 pos_playlist_3 neg_playlist_3, SVM_Model.train 1.0 pos_playlist_3 neg_playlist_3)
  |> fun (svm_1, svm_2) -> SVM_Model.equal svm_1 svm_2);
;;

let test_svm_equal _ =
  assert_bool "Model 1 not equal to itself" @@ SVM_Model.equal svm_1 svm_1;
  assert_bool "Model 2 not equal to itself" @@ SVM_Model.equal svm_2 svm_2;
  assert_bool "Model 3 not equal to itself" @@ SVM_Model.equal svm_3 svm_3;
;;

let test_svm_save_load _ =
  let filename_1 = "./model_1" in
  assert_bool "Error in saving or loading Model 1" 
  @@ (SVM_Model.save svm_1 filename_1; SVM_Model.equal svm_1 @@ SVM_Model.load filename_1);
  let filename_2 = "./model_2" in
  assert_bool "Error in saving or loading Model 2" 
  @@ (SVM_Model.save svm_2 filename_2; SVM_Model.equal svm_2 @@ SVM_Model.load filename_2);
  let filename_3 = "./model_3" in
  assert_bool "Error in saving or loading Model 3" 
  @@ (SVM_Model.save svm_3 filename_3; SVM_Model.equal svm_3 @@ SVM_Model.load filename_3);
  let bad_filename = "./bad"
  in let bad_load = fun () -> let f = Stdio.Out_channel.create bad_filename 
       in Stdio.Out_channel.output_string f "hi"; Stdio.Out_channel.flush f;
       Stdio.Out_channel.close f; (SVM_Model.load bad_filename |> fun _ -> ()) 
  in assert_raises (Failure "improper file formatting") bad_load;  
;;

let test_svm_classes _ =
  assert_equal (SVM_Model.classes svm_1) ("Positive Playlist 1", "Negative Playlist 1");
  assert_equal (SVM_Model.classes svm_2) ("Positive Playlist 2", "Negative Playlist 2");
  assert_equal (SVM_Model.classes svm_3) ("Positive Playlist 3", "Negative Playlist 3");
;;

let svm_tests =
  "SVM" >: test_list [
    "Classes" >:: test_svm_classes;
    "Equal" >:: test_svm_equal;
    "Save and Load" >:: test_svm_save_load;
    "Train and Predict" >:: test_svm_train_predict;
  ]

let machine_learning_tests =
  "Machine Learning" >: test_list [
    "Classification" >:: test_classify;
    "Test" >:: test_test;
    "Pretty Confusion" >:: test_pretty_confusion;
    "Accuracy" >:: test_acc;
    "F1 Score" >:: test_f1;
  ]

let series =
  "Final Project Tests" >::: [
    svm_tests;
    machine_learning_tests;
  ]

let () = 
  run_test_tt_main series