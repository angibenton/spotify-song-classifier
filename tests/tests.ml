open Core
open OUnit2
open Machine_learning
open Svm
let pretty_confusion_1 = "               actual  
               pos neg 
              ---------
          pos | 5 | 0 |
predicted     ---------
          neg | 6 | 4 |
              ---------
";;
let pretty_confusion_2 = "                 actual     
                 pos   neg   
              ---------------
          pos | 5543 |    5 |
predicted     ---------------
          neg |   26 |  314 |
              ---------------
";;

let test_pretty_confusion _ = 
  let cm = SVM_Classification.pretty_confusion {tp = 5; fp = 0; fn = 6; tn = 4}
  in let (f,g) = (Stdio.Out_channel.create "./testfile.txt", Stdio.Out_channel.create "./testfile2.txt")  in Stdio.Out_channel.output_string f pretty_confusion_1; Stdio.Out_channel.flush f;
  Stdio.Out_channel.close f; Stdio.Out_channel.output_string g cm; Stdio.Out_channel.flush g;
  Stdio.Out_channel.close g;
  assert_equal cm pretty_confusion_1;
;;

let test_classes _ =
  assert_equal (SVM_Model.train 1. {features_matrix = (Np.matrixf[|[|5.|]|]); pid = "1"; name= "hi"} {features_matrix = (Np.matrixf[|[|6.|]|]); pid = "2"; name = "6"} |> SVM_Model.classes ) ("hi", "6");
;;

let svm_tests =
  "SVM" >: test_list [
    "Flip" >:: test_classes;
  ]

let machine_learning_tests =
  "Machine Learning" >: test_list [
    "Pretty Confusion" >:: test_pretty_confusion;
  ]

let series =
  "Final Project Tests" >::: [
    svm_tests;
    machine_learning_tests;
  ]

let () = 
  run_test_tt_main series