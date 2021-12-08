open Core
open OUnit2
open Machine_learning
open Svm
open Spotify

let float_equals_epsilon = fun f1 f2 -> Float.(-) f1 f2 
                                        |> Float.abs 
                                        |>  Float.(>) 0.001;;

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

let pos_song_1_1 = {name = "Positive Song 1"; sid = "1";features_vector = Np.Ndarray.vectorf [|1.|]}
let pos_song_1_2 = {name = "Positive Song 2"; sid = "2";features_vector = Np.Ndarray.vectorf [|0.5|]}

let pos_song_2 = {name = "Positive Song 1"; sid = "1";features_vector = Np.Ndarray.vectorf [| 10.; 10. |]}
let neg_song_2 = {name = "Positive Song 1"; sid = "1";features_vector = Np.Ndarray.vectorf [| -10.; -10. |]}

let pos_playlist_2 = {name = "the first one"; pid = "123"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array pos_song_2.features_vector  |])};;
let neg_playlist_2 = {name = "the second one"; pid = "124"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array neg_song_2.features_vector |])};;

let pos_playlist_3 = {name = "the first one"; pid = "123"; 
                      features_matrix = (Np.matrixf [| [| 0.; 5. |];  |])};;
let neg_playlist_3 = {name = "the second one"; pid = "124"; 
                      features_matrix = (Np.matrixf [| [| 0.; 3. |]; |])};;

(*
let test_svm_predict_score _ = 
  assert_bool "Model 1 Test 1" @@ float_equals_epsilon (1.0) 
  @@ SVM_Model.predict_score svm_1.hyperplane svm_1.intercept pos_song_1_1.features_vector;
  assert_bool "Model 1 Test 1" @@ float_equals_epsilon (0.5) 
  @@ SVM_Model.predict_score svm_1.hyperplane svm_1.intercept pos_song_1_2.features_vector;
;;

let test_svm_predict _ = 
  assert_bool "Model 1 Test 1" @@ SVM_Model.predict svm_1 @@ pos_song_1_1.features_vector;
  (*assert_bool "Model 1 Test 2" @@ SVM_Model.predict svm_1 @@ Np.Ndarray.vectorf [|-2.|];*)
;;*)
(*
let test_classify _ = 
  assert_equal "Positive" @@ SVM_Classification.classify svm_1 pos_song_1_1;
;;
*)
let test_svm_train _ = 
  assert_bool "Model 2 misclassifies its positive song" 
    (SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2 
     |> fun svm -> SVM_Model.predict svm @@ pos_song_2.features_vector );
  assert_bool "SVM should train deterministically" (
    (SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2, SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2)
    |> fun (svm_1, svm_2) -> SVM_Model.equal svm_1 svm_2);
  assert_bool "Model 3" 
    (SVM_Model.train 1.0 pos_playlist_3 neg_playlist_3
     |> fun svm -> SVM_Model.predict svm @@ Np.Ndarray.vectorf [|0.; 5.;|]);
  assert_bool "SVM should train deterministically" (
    (SVM_Model.train 1.0 pos_playlist_3 neg_playlist_3, SVM_Model.train 1.0 pos_playlist_3 neg_playlist_3)
    |> fun (svm_1, svm_2) -> SVM_Model.equal svm_1 svm_2);
;;

let test_svm_equal _ =
  assert_bool "Model 2 not equal to itself" 
    (SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2 
     |> fun svm -> SVM_Model.equal svm svm);
  assert_bool "Model 3 not equal to itself" 
    (SVM_Model.train 1.0 pos_playlist_3 neg_playlist_3
     |> fun svm -> SVM_Model.equal svm svm);
;;

let test_svm_save_load _ =
  let filename_2 = "./model_2" in
  assert_bool "Error in saving or loading Model 2" 
  @@ let orig = SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2 
  in SVM_Model.save orig filename_2; SVM_Model.equal orig @@ SVM_Model.load filename_2;
;;

let test_svm_classes _ =
  assert_equal (SVM_Model.train 1. {features_matrix = (Np.matrixf[|[|5.|]|]); pid = "1"; name= "hi"}
                  {features_matrix = (Np.matrixf[|[|6.|]|]); pid = "2"; name = "6"} |> SVM_Model.classes ) ("hi", "6");
;;

let svm_tests =
  "SVM" >: test_list [
    "Classes" >:: test_svm_classes;
    "Equal" >:: test_svm_equal;
    "Save and Load" >:: test_svm_save_load;

    (*"Train" >:: test_svm_train;*)
    (*"Predict" >:: test_svm_predict;
      "Predict Score" >:: test_svm_predict_score;*)
  ]

let machine_learning_tests =
  "Machine Learning" >: test_list [
    (*"Classification" >:: test_classify;*)
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