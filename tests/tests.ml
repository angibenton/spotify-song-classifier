open Core
open OUnit2
open Machine_learning
open Svm
(*module Np = Np.Numpy*)
open Spotify

let float_equals_epsilon = fun f1 f2 -> Float.(-) f1 f2 
                                        |> Float.abs 
                                        |>  Float.(>) 0.001;;

let cm_1 = {tp = 5; fp = 0; fn = 6; tn = 4};;
let cm_2 = {tp = 5543; fp = 5; fn = 5852; tn = 314};;
let cm_3 = {tp = 9; fp = 588097; fn = 907; tn = 78954};;
let pretty_confusion_1 = "               actual  
               pos neg 
              ---------
          pos | 5 | 0 |
predicted     ---------
          neg | 6 | 4 |
              ---------
";;
let pretty_confusion_2 = "                  actual     
                 pos   neg   
              ---------------
          pos | 5543 |    5 |
predicted     ---------------
          neg | 5852 |  314 |
              ---------------
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
  assert_bool "Confusion Matrix 3" (float_equals_epsilon 0.6 
                                    @@ SVM_Classification.accuracy cm_1);
;;
assert_bool "Confusion Matrix 3" (float_equals_epsilon 0.5 
                                  @@ SVM_Classification.accuracy cm_2);
;;
assert_bool "Confusion Matrix 3" (float_equals_epsilon 0.1182 
                                  @@ SVM_Classification.accuracy cm_3);
;;

let pos_playlist_1 = {name = "the first one"; pid = "123"; 
                      features_matrix = (Np.matrixf [| [| 10.; 10. |];  |])};;
let neg_playlist_1 = {name = "the second one"; pid = "124"; 
                      features_matrix = (Np.matrixf [| [| -10.; -10. |]; |])};;
let hyperplane_1 = [|-0.0499; -0.0499;|];;
let intercept_1 = 0.;;

let pos_playlist_2 = {name = "the first one"; pid = "123"; 
                      features_matrix = (Np.matrixf [| [| 0.; 5. |];  |])};;
let neg_playlist_2 = {name = "the second one"; pid = "124"; 
                      features_matrix = (Np.matrixf [| [| 0.; 3. |]; |])};;
let hyperplane_2 = [|0.; -0.2247;|];;
let intercept_2 = -0.7191;;

let test_svm_train _ = 
  assert_bool "Model 1" 
    (SVM_Model.train 1.0 pos_playlist_1 neg_playlist_1 
     |> fun {hyperplane; intercept; _} -> Np.Ndarray.to_float_array hyperplane 
                                           |>fun model_hyperplane -> Array.fold ~init:0 ~f:(fun _ num -> Stdio.printf "elem: %f" num; 0) model_hyperplane |> fun _ -> (float_equals_epsilon intercept intercept_1) && Array.equal (fun num1 num2 -> float_equals_epsilon num1 num2) model_hyperplane hyperplane_1);
  assert_bool "Model 2" 
    (SVM_Model.train 1.0 pos_playlist_2 neg_playlist_2
     |> fun {hyperplane; intercept; _} -> Printf.printf "intercept: %f" intercept; Np.Ndarray.to_float_array hyperplane 
                                          |> fun model_hyperplane -> Array.fold ~init:0 ~f:(fun _ num -> Stdio.printf "elem: %f" num; 0) model_hyperplane |> fun _ -> (float_equals_epsilon intercept intercept_2) && Array.equal (fun num1 num2 -> float_equals_epsilon num1 num2) model_hyperplane hyperplane_2);
;;
let test_svm_classes _ =
  assert_equal (SVM_Model.train 1. {features_matrix = (Np.matrixf[|[|5.|]|]); pid = "1"; name= "hi"}
                  {features_matrix = (Np.matrixf[|[|6.|]|]); pid = "2"; name = "6"} |> SVM_Model.classes ) ("hi", "6");
;;

let svm_tests =
  "SVM" >: test_list [
    "Classes" >:: test_svm_classes;
    "Train" >:: test_svm_train;
  ]

let machine_learning_tests =
  "Machine Learning" >: test_list [
    "Pretty Confusion" >:: test_pretty_confusion;
    "Accuracy" >:: test_acc;
  ]

let test_get_token _ = 
  let test_monadic = 
    let%lwt _token = Spotify_api.get_new_api_token () in
    printf "bullshit";
    Lwt.return true; 
  in
  assert_equal true @@ Lwt_main.run test_monadic;  
;;

let spotify_api_tests = 
    "Spotify API Tests" >: test_list [
      "Get Token" >:: test_get_token; 
    ]


let series =
  "Final Project Tests" >::: [
    svm_tests;
    machine_learning_tests;
  ]

let () = 
  run_test_tt_main series