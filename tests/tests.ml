open Core
open OUnit2
open Machine_learning
open Svm

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
  assert_bool "Confusion Matrix 3" (Float.(-) (SVM_Classification.accuracy cm_1) 
                                      0.6 |> Float.abs |> Float.(>) 0.001);
;;
assert_bool "Confusion Matrix 3" (Float.(-) (SVM_Classification.accuracy cm_2) 
                                    0.5 |> Float.abs |> Float.(>) 0.001);
;;
assert_bool "Confusion Matrix 3" (Float.(-) (SVM_Classification.accuracy cm_3) 
                                    0.1182 |> Float.abs |> Float.(>) 0.001);
;;

let test_svm_classes _ =
  assert_equal (SVM_Model.train 1. {features_matrix = (Np.matrixf[|[|5.|]|]); pid = "1"; name= "hi"}
                  {features_matrix = (Np.matrixf[|[|6.|]|]); pid = "2"; name = "6"} |> SVM_Model.classes ) ("hi", "6");
;;

let svm_tests =
  "SVM" >: test_list [
    "Classes" >:: test_svm_classes;
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