open Core
open OUnit2
open OUnitLwt
open Machine_learning
open Svm
open Spotify
open Numpy_helper

let float_equals_epsilon_custom 
  = fun f1 f2 eps -> Float.(-) f1 f2 |> Float.abs |>  Float.(>) eps;;

let float_equals_epsilon
  = fun f1 f2 -> Float.(-) f1 f2 |> Float.abs |>  Float.(>) 0.001;;

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

(* Artificial Songs, Playlists, Hyperparameters, SVM's, etc. for tests *)

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
                  features_vector = Np.reshape ~newshape:[1; 2] 
                    @@  Np.Ndarray.vectorf [| -10.; -10. |]}

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

let pos_song_4_1 = {name = "Positive Song 1"; sid = "3";
                    features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                        [| 0.; 5.; 83.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 5.2; 89.1|]};;

let pos_song_4_2 = {name = "Positive Song 1"; sid = "3";
                    features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                        [| 0.; 87.2; 11.2; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.1; 2.2|]};;

let pos_song_4_3 = {name = "Positive Song 1"; sid = "3";
                    features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                        [| 0.; -1.9; 3.1; 3.2; 90.; 0.; 0.; 0.; 0.; 0.; 0.; 35.4; 0.5|]};;

let neg_song_4 = {name = "Negative Song 3"; sid = "3";
                  features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                      [| 0.; 42.; 0.; 87.; 0.; 21.; 0.; 67.; 0.; 0.; 0.; 90.2; -8.9|]};;

let pos_song_5 = {name = "Negative Song 3"; sid = "3";
                  features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                      [| 91.4; 3.2; 5.6; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; -89.; 2.13|]};;

let neg_song_5_1 = {name = "Negative Song 3"; sid = "3";
                    features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                        [| 0.; 8.2; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 53.2; 2.1;|]};;

let neg_song_5_2 = {name = "Negative Song 3"; sid = "3";
                    features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                        [| 3.2; 35.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 2.; 0.|]};;

let neg_song_5_3 = {name = "Negative Song 3"; sid = "3";
                    features_vector = Np.reshape ~newshape:[1; 13] @@ Np.Ndarray.vectorf 
                        [| 9.2; 2.3; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 12.; 0.|]};;

let pos_playlist_4 = {name = "Positive Playlist 4"; pid = "123"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array pos_song_4_1.features_vector; 
                                                       Np.Ndarray.to_float_array pos_song_4_2.features_vector; 
                                                       Np.Ndarray.to_float_array pos_song_4_3.features_vector; |])};;
let neg_playlist_4 = {name = "Negative Playlist 4"; pid = "124"; 
                      features_matrix = (Np.matrixf [|Np.Ndarray.to_float_array neg_song_4.features_vector |])};;

let pos_playlist_5 = {name = "Positive Playlist 5"; pid = "123"; 
                      features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array pos_song_5.features_vector |])};;
let neg_playlist_5 = {name = "Negative Playlist 5"; pid = "124"; 
                      features_matrix = (Np.matrixf [|Np.Ndarray.to_float_array neg_song_5_1.features_vector; 
                                                      Np.Ndarray.to_float_array neg_song_5_2.features_vector;  
                                                      Np.Ndarray.to_float_array neg_song_5_3.features_vector |])};;

let pos_playlist_big = {name = "Positive Playlist Big"; pid = "123"; 
                        features_matrix = (Np.matrixf [| Np.Ndarray.to_float_array pos_song_4_1.features_vector; 
                                                         Np.Ndarray.to_float_array pos_song_4_2.features_vector; 
                                                         Np.Ndarray.to_float_array pos_song_4_3.features_vector; 
                                                         Np.Ndarray.to_float_array pos_song_5.features_vector; 
                                                         Np.Ndarray.to_float_array pos_song_3.features_vector |])};;
let neg_playlist_big = {name = "Negative Playlist Big"; pid = "124"; 
                        features_matrix = (Np.matrixf [|Np.Ndarray.to_float_array neg_song_5_1.features_vector; 
                                                        Np.Ndarray.to_float_array neg_song_5_2.features_vector; 
                                                        Np.Ndarray.to_float_array neg_song_5_3.features_vector; 
                                                        Np.Ndarray.to_float_array neg_song_4.features_vector; 
                                                        Np.Ndarray.to_float_array neg_song_3.features_vector |])};;

let standard_hyper : SVM_Model.hyperparameters = {reg = 1.0; shift = Np.empty [0]; scale = Np.empty [0]};;
let huge_hyper : SVM_Model.hyperparameters = {reg = 1000.0; shift = Np.empty [0]; scale = Np.empty [0]};;
let preprocess_hyper : SVM_Model.hyperparameters = {reg = 1.0; 
                                                    shift = Np.Ndarray.of_float_list [0.;0.;0.;0.;0.;0.;0.;0.;0.;0.;0.;0.;0.;]; 
                                                    scale = Np.Ndarray.of_float_list [1.;1.;1.;1.;1.;1.;1.;1.;1.;1.;1.;1.;1.;]};;


let svm_1 = SVM_Model.train standard_hyper pos_playlist_1 neg_playlist_1
let svm_2 = SVM_Model.train standard_hyper pos_playlist_2 neg_playlist_2
let svm_3 = SVM_Model.train standard_hyper  pos_playlist_3 neg_playlist_3
let svm_4 = SVM_Model.train standard_hyper pos_playlist_4 neg_playlist_4
let svm_4_reg = SVM_Model.train huge_hyper pos_playlist_3 neg_playlist_3
let svm_preprocess = SVM_Model.train preprocess_hyper pos_playlist_3 neg_playlist_3

let test_classify _ = 
  assert_equal "Positive Playlist 1" @@ SVM_Classification.classify svm_1 pos_song_1;
  assert_equal "Negative Playlist 1" @@ SVM_Classification.classify svm_1 neg_song_1;
  assert_equal "Positive Playlist 2" @@ SVM_Classification.classify svm_2 pos_song_2;
  assert_equal "Negative Playlist 2" @@ SVM_Classification.classify svm_2 neg_song_2;
  assert_equal "Positive Playlist 3" @@ SVM_Classification.classify svm_3 pos_song_3;
  assert_equal "Negative Playlist 3" @@ SVM_Classification.classify svm_3 neg_song_3;
  assert_equal "Positive Playlist 3" @@ SVM_Classification.classify svm_preprocess pos_song_3;
  assert_equal "Negative Playlist 3" @@ SVM_Classification.classify svm_preprocess neg_song_3;
;;

let test_tune _ =
  let chosen = SVM_Classification.tune [svm_4_reg; svm_4; svm_preprocess] pos_playlist_4 neg_playlist_4 SVM_Classification.accuracy
  in assert_equal "Positive Playlist 4" @@ SVM_Classification.classify chosen pos_song_4_1;
  assert_equal "Negative Playlist 4" @@ SVM_Classification.classify chosen neg_song_4;
  assert_raises (Failure "No models provided") 
    (fun () -> SVM_Classification.tune [] pos_playlist_4 neg_playlist_4 SVM_Classification.accuracy)
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
    (SVM_Model.train standard_hyper pos_playlist_1 neg_playlist_1, SVM_Model.train standard_hyper pos_playlist_1 neg_playlist_1)
    |> fun (svm_1, svm_2) -> SVM_Model.equal svm_1 svm_2);
  assert_bool "Model 2 misclassifies its positive song" 
  @@ SVM_Model.predict svm_2 @@ pos_song_2.features_vector;
  assert_bool "Model 2 misclassifies its negative song" 
  @@ not @@ SVM_Model.predict svm_2 @@ neg_song_2.features_vector;
  assert_bool "SVM should train deterministically" (
    (SVM_Model.train standard_hyper pos_playlist_2 neg_playlist_2, SVM_Model.train standard_hyper pos_playlist_2 neg_playlist_2)
    |> fun (svm_1, svm_2) -> SVM_Model.equal svm_1 svm_2);
  assert_bool "Model 3 misclassifies its positive song" 
  @@ SVM_Model.predict svm_3 @@ pos_song_3.features_vector;
  assert_bool "Model 3 misclassifies its negative song" 
  @@ not @@ SVM_Model.predict svm_3 @@ neg_song_3.features_vector;
  assert_bool "SVM should train deterministically" (
    (SVM_Model.train standard_hyper pos_playlist_3 neg_playlist_3, SVM_Model.train standard_hyper pos_playlist_3 neg_playlist_3)
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

let test_balance _ = 
  assert_equal (Np.Ndarray.to_float_array pos_playlist_3.features_matrix) 
    (SVM_Classification.balance_classes (pos_playlist_3, neg_playlist_3) 
     |> fun (new_pos, _) -> Np.Ndarray.to_float_array new_pos.features_matrix);
  assert_bool "Balanced playlists still balanced" 
    (SVM_Classification.balance_classes (pos_playlist_1, neg_playlist_1) 
     |> fun (new_pos, new_neg) -> Np.size new_pos.features_matrix = Np.size new_neg.features_matrix);
  assert_bool "Songs not subsampled in imbalance" 
    (SVM_Classification.balance_classes (pos_playlist_4, neg_playlist_4) 
     |> fun (new_pos, new_neg) -> Np.size new_pos.features_matrix =  Np.size new_neg.features_matrix);
;;

let test_vector_equal _ =
  assert_bool "Vector does not equal itself" @@ vector_equal pos_song_1.features_vector pos_song_1.features_vector;
  assert_bool "Vector does not equal itself" @@ vector_equal neg_song_1.features_vector neg_song_1.features_vector;
  assert_bool "Vector does not equal itself" @@ vector_equal pos_song_4_1.features_vector pos_song_4_1.features_vector;
;;

let test_matrix_equal _ =
  assert_bool "Matrix does not equal itself" @@ matrix_equal pos_playlist_1.features_matrix pos_playlist_1.features_matrix;
  assert_bool "Matrix does not equal itself" @@ matrix_equal pos_playlist_3.features_matrix pos_playlist_3.features_matrix;
  assert_bool "Matrix does not equal itself" @@ matrix_equal pos_playlist_4.features_matrix pos_playlist_4.features_matrix;
;;

let test_matrix_of_vector_list _ =
  assert_bool "Singleton matrix created from list not as expected" 
    (matrix_equal pos_playlist_1.features_matrix @@ matrix_of_vector_list [pos_song_1.features_vector]); 
  assert_bool "Real matrix created from list not as expected" 
    (matrix_equal pos_playlist_4.features_matrix 
     @@ matrix_of_vector_list [pos_song_4_1.features_vector; 
                               pos_song_4_2.features_vector; pos_song_4_3.features_vector]); 
;;

let single = Np.Ndarray.of_float_list [0.; 1.; 2.; 3.;]

let added = Np.Ndarray.of_float_list [1.; 2.; 3.; 4.;]

let doubled = Np.Ndarray.of_float_list [0.; 2.; 4.; 6.;]

let neg = Np.Ndarray.of_float_list [-1.; -2.; -3.; -4.;]

let combined = matrix_of_vector_list [single;added;doubled;neg;]

let test_vector_map _ =
  assert_bool "No operation mapping" @@ vector_equal single @@ map_vector single (fun x -> x);
  assert_bool "Added 1 to each" @@ vector_equal added @@ map_vector single (fun x -> Float.(+) x 1.);
  assert_bool "Doubled vector entries" @@ vector_equal doubled @@ map_vector single (fun x -> Float.( * ) x 2.);
  assert_bool "Negative vector entries" @@ vector_equal neg @@ map_vector single (fun x -> Float.( * ) (-1.0) @@ Float.(+) x 1.);
;;

let test_vector_min _ = 
  assert_bool "Found non-minimum element in vector 1" @@ float_equals_epsilon 0. @@ vec_min single;
  assert_bool "Found non-minimum element in vector 2" @@ float_equals_epsilon 0. @@ vec_min doubled;
  assert_bool "Found non-minimum element in vector 3" @@ float_equals_epsilon 1. @@ vec_min added;
  assert_bool "Found non-minimum element in vector 4" @@ float_equals_epsilon (-4.0) @@ vec_min neg;
;;

let test_vector_max _ = 
  assert_bool "Found non-maximum element in vector 1" @@ float_equals_epsilon 3. @@ vec_max single;
  assert_bool "Found non-maximum element in vector 2" @@ float_equals_epsilon 6. @@ vec_max doubled;
  assert_bool "Found non-maximum element in vector 3" @@ float_equals_epsilon 4. @@ vec_max added;
  assert_bool "Found non-maximum element in vector 4" @@ float_equals_epsilon (-1.0) @@ vec_max neg;
;;

let test_vector_mean _ = 
  assert_bool "Incorrect mean 1" @@ float_equals_epsilon 1.5 @@ vec_mean single;
  assert_bool "Incorrect mean 2" @@ float_equals_epsilon 3. @@ vec_mean doubled;
  assert_bool "Incorrect mean 3" @@ float_equals_epsilon 2.5 @@ vec_mean added;
  assert_bool "Incorrect mean 4" @@ float_equals_epsilon (-3.0) @@ vec_mean neg;
;;

let test_vector_std _ = 
  assert_bool "Incorrect Std 1" @@ float_equals_epsilon 1.118 @@ vec_std single @@ vec_mean single;
  assert_bool "Incorrect Std 2" @@ float_equals_epsilon 2.2361 @@ vec_std doubled @@ vec_mean doubled;
  assert_bool "Incorrect Std 3" @@ float_equals_epsilon 1.118 @@ vec_std added @@ vec_mean added;
  assert_bool "Incorrect Std 4" @@ float_equals_epsilon 1.118 @@ vec_std neg @@ vec_mean neg;
;;

let test_vector_to_string _ = 
  assert_equal " 0. 1. 2. 3." @@ vector_to_string single;
  assert_equal " 1. 2. 3. 4." @@ vector_to_string added;
  assert_equal " 0. 2. 4. 6." @@ vector_to_string doubled;
  assert_equal " -1. -2. -3. -4." @@ vector_to_string neg;
;;

let test_normalize _ = 
  assert_bool "Not within correct range" (SVM_Classification.normalize (pos_playlist_big, neg_playlist_big) 
                                          |> fun (new_pos, _, _) 
                                          -> List.for_all ~f:(fun col -> 
                                              Array.for_all ~f:(fun elem 
                                                                 -> ((Float.(>=) 1.0 elem) 
                                                                     && (Float.(>=) elem 0.0))) 
                                              @@ Np.Ndarray.to_float_array col) 
                                            (matrix_columns_to_vector_list new_pos.features_matrix)); 
;;

let test_standardize _ = 
  assert_bool "Not within correct distribution" (SVM_Classification.standardize (pos_playlist_big, neg_playlist_big) 
                                                 |> fun (new_pos, _, _) 
                                                 -> List.for_all ~f:(fun col ->  
                                                     ((float_equals_epsilon_custom 0.0 (vec_mean col) 0.5) 
                                                      && (vec_std col @@ vec_mean col 
                                                          |> fun std -> float_equals_epsilon_custom 1.0 std 0.5 
                                                                        || float_equals_epsilon_custom 0.0 std 0.5)))
                                                   (matrix_columns_to_vector_list new_pos.features_matrix)); 
;;

let test_randomize _ =
  assert_equal (Np.size pos_playlist_big.features_matrix) (SVM_Classification.randomize pos_playlist_big 
                                                           |> fun p -> Np.size p.features_matrix);
;;

let test_split _ = 
  let (pos, neg, preprocess) = SVM_Classification.normalize (pos_playlist_big, neg_playlist_big)
  in assert_raises (Failure "Validation split size too small") (fun () -> SVM_Classification.split (pos, neg) 0.0 0.25 preprocess);
  assert_raises (Failure "Test split size too small") (fun () -> SVM_Classification.split (pos, neg) 0.25 0.0 preprocess);
  assert_raises (Failure "Train split size too small (validation + test too big)") 
    (fun () -> SVM_Classification.split (pos, neg) 0.25 0.75 preprocess);
  assert_bool "Normal Split with normalize" 
    (SVM_Classification.normalize (pos_playlist_big, neg_playlist_big) 
     |> fun (pos, neg, preprocess) -> SVM_Classification.split (pos, neg) 0.25 0.25 preprocess 
                                      |> fun {pos_train; pos_valid; pos_test; _} 
                                      -> 2 = (List.length @@ matrix_rows_to_vector_list pos_train.features_matrix) 
                                         && 1 = (List.length @@ matrix_rows_to_vector_list pos_valid.features_matrix) 
                                         && 2 = (List.length @@ matrix_rows_to_vector_list pos_test.features_matrix));
  assert_bool "Normal Split with standardize" 
    (SVM_Classification.standardize (pos_playlist_big, neg_playlist_big) 
     |> fun (pos, neg, preprocess) -> SVM_Classification.split (pos, neg) 0.25 0.25 preprocess 
                                      |> fun {pos_train; pos_valid; pos_test; _} 
                                      -> 2 = (List.length @@ matrix_rows_to_vector_list pos_train.features_matrix) 
                                         && 1 = (List.length @@ matrix_rows_to_vector_list pos_valid.features_matrix) 
                                         && 2 = (List.length @@ matrix_rows_to_vector_list pos_test.features_matrix));
  assert_bool "Normal Split without preprocessing" 
    ((pos_playlist_big, neg_playlist_big, []) 
     |> fun (pos, neg, preprocess) 
     -> SVM_Classification.split (pos, neg) 0.25 0.25 preprocess 
        |> fun {pos_train; pos_valid; pos_test;_} 
        -> 2 = (List.length @@ matrix_rows_to_vector_list pos_train.features_matrix) 
           && 1 = (List.length @@ matrix_rows_to_vector_list pos_valid.features_matrix) 
           && 2 = (List.length @@ matrix_rows_to_vector_list pos_test.features_matrix));
;;

let test_save_load_dataset _ =
  assert_bool "Save and load normalized dataset" 
    (SVM_Classification.normalize (pos_playlist_big, neg_playlist_big) 
     |> fun (pos, neg, preprocess) 
     -> SVM_Classification.split (pos, neg) 0.25 0.25 preprocess 
        |> fun d -> let unique = Float.to_string @@ Unix.time () 
        in SVM_Classification.save_dataset d @@ "testing" ^ unique; 
        SVM_Classification.load_dataset @@ "testing" ^ unique |> 
        fun {pos_train; pos_valid; pos_test; _} 
        -> (matrix_equal pos_train.features_matrix d.pos_train.features_matrix 
            && matrix_equal pos_valid.features_matrix d.pos_valid.features_matrix 
            && matrix_equal pos_test.features_matrix d.pos_test.features_matrix));
  assert_bool "Save and load standardized dataset" 
    (SVM_Classification.standardize (pos_playlist_big, neg_playlist_big) |>
     fun (pos, neg, preprocess) -> 
     SVM_Classification.split (pos, neg) 0.25 0.25 preprocess |>
     fun d -> let unique = Float.to_string @@ Float.(+) 1.0 
                @@ Unix.time () in SVM_Classification.save_dataset d 
     @@ "testing" ^ unique; SVM_Classification.load_dataset 
     @@ "testing" ^ unique |> fun {pos_train; pos_valid; pos_test; _} 
                            -> (matrix_equal pos_train.features_matrix d.pos_train.features_matrix 
                                && matrix_equal pos_valid.features_matrix d.pos_valid.features_matrix 
                                && matrix_equal pos_test.features_matrix d.pos_test.features_matrix));
  assert_bool "Save and load non-preprocessed dataset" 
    ((pos_playlist_big, neg_playlist_big, []) 
     |> fun (pos, neg, preprocess) 
     -> SVM_Classification.split (pos, neg) 0.25 0.25 preprocess 
        |> fun d -> let unique = Float.to_string @@ Float.(+) 2.0 @@ Unix.time () 
        in SVM_Classification.save_dataset d @@ "testing" ^ unique; SVM_Classification.load_dataset 
        @@ "testing" ^ unique 
                                                                    |> fun {pos_train; pos_valid; pos_test; _} 
                                                                    -> (matrix_equal pos_train.features_matrix d.pos_train.features_matrix 
                                                                        && matrix_equal pos_valid.features_matrix d.pos_valid.features_matrix 
                                                                        && matrix_equal pos_test.features_matrix d.pos_test.features_matrix));
;;


let numpy_tests =
  "Numpy" >: test_list [
    "Vector Equal" >:: test_vector_equal;
    "Matrix Equal" >:: test_matrix_equal;
    "Matrix of Vectors List" >:: test_matrix_of_vector_list;
    "Map Vectors" >:: test_vector_map;
    "Vector Min" >:: test_vector_min;
    "Vector Max" >:: test_vector_max;
    "Vector Mean" >:: test_vector_max;
    "Vector Std" >:: test_vector_std;
    "Vector To String" >:: test_vector_to_string;
  ]

let svm_tests =
  "SVM" >: test_list [
    "Classes" >:: test_svm_classes;
    "Equal" >:: test_svm_equal;
    "Save and Load" >:: test_svm_save_load;
    "Train and Predict" >:: test_svm_train_predict;
    "Tune" >:: test_tune;
  ]

let machine_learning_tests =
  "Machine Learning" >: test_list [
    "Classification" >:: test_classify;
    "Test" >:: test_test;
    "Pretty Confusion" >:: test_pretty_confusion;
    "Accuracy" >:: test_acc;
    "F1 Score" >:: test_f1;
    "Class Balancing" >:: test_balance;
    "Normalization" >:: test_normalize;
    "Standardization" >:: test_standardize;
    "Randomization" >:: test_randomize;
    "Split Dataset" >:: test_split;
    "Save and Load Dataset" >:: test_save_load_dataset;
  ]

(* ----- Spotify Tests - Asynchronous!! Tests return unit Lwt.t rather than unit ------ *)

(* test data *)
let number_of_features = 13;;
let test_playlist_id = "01keRwFGHF7Rw1wnPwbyB1";;
let test_song_id_vienna = "4U45aEWtQhrm8A5mxPaFZ7";;
let test_song_id_chain = "5e9TFTbltYBg2xThimr0rU";;
let test_song_id_wish = "1HzDhHApjdjXPLHF6GGYhu";;
let test_song_id_skylines = "3VqJUdav5hRAEAvppYIVnT";;
let expired_token = "BQCeq-iZTX3AWwOhNVt9fM3vZBicMy8pVSINCAjgJ-Yge4PnF-OPae-SMVLJdzk-YiM10yFxtnZrntcxo2w";;
let ill_formed_token = "MyToken123";;
let ill_formed_song_id = "MySongId123";;
let ill_formed_playlist_id = "MySongId123";;

(* check if token is well-formed *)
(* other spotify api tests implicitly check if the access token is accepted by the server *)
let test_get_token _ = 
  let%lwt token = get_new_api_token () in
  assert_equal 83 @@ String.length token;
  assert_equal token @@ String.filter ~f:(fun c -> Char.is_alphanum c || Char.(=) c '-' || Char.(=) c '_') token;
  Lwt.return (); 
;;

(* check song contains expected data *)
let test_get_song _ = 
  let%lwt token = get_new_api_token () in
  let%lwt chain = song_of_id test_song_id_chain token in
  let%lwt skylines = song_of_id test_song_id_skylines token in
  assert_equal "The Chain - 2004 Remaster" @@ chain.name;
  assert_equal "Skylines and Turnstiles" @@ skylines.name;
  assert_equal test_song_id_chain @@ chain.sid;
  assert_equal test_song_id_skylines @@ skylines.sid;
  assert_equal [|1; number_of_features;|] @@ Np.shape chain.features_vector;
  assert_equal [|1; number_of_features;|] @@ Np.shape skylines.features_vector;
  Lwt.return (); 
;;

(* check playlist contains expected data *)
let test_get_playlist _ = 
  let%lwt token = get_new_api_token () in
  let%lwt test_playlist = playlist_of_id test_playlist_id token in
  assert_equal "TestPlaylist" @@ test_playlist.name;
  assert_equal test_playlist_id @@ test_playlist.pid;
  (* this is my own playlist, definitely has 3 songs *)
  assert_equal [|3; number_of_features;|] @@ Np.shape test_playlist.features_matrix;
  Lwt.return (); 
;;

(* helper for testing if an asynchronous function would raise an exception with a given  *)
let assert_promise_would_raise_exception (promise: 'a Lwt.t) (message: string): unit Lwt.t = 
  Lwt.catch (fun _ -> let%lwt _ = promise in Lwt.return ();)
  @@ (fun exn -> match exn with 
      | Stdlib.Failure s -> assert_equal s @@ message; Lwt.return (); 
      | _ -> assert false;
    )
;;

(* expired and bad tokens used for song_of_id and playlist_of_id *)
let test_bad_token_exceptions _ = 
  let%lwt () = assert_promise_would_raise_exception (song_of_id test_song_id_vienna ill_formed_token) 
      ("request features for song (id = " ^ test_song_id_vienna ^ ") failed: 401 Invalid access token") in  
  let%lwt () = assert_promise_would_raise_exception (song_of_id test_song_id_vienna expired_token) 
      ("request features for song (id = " ^ test_song_id_vienna ^ ") failed: 401 The access token expired") in 
  let%lwt () = assert_promise_would_raise_exception (playlist_of_id test_playlist_id ill_formed_token)
      ("request metadata for playlist (id = " ^ test_playlist_id ^ ") failed: 401 Invalid access token") in 
  let%lwt () = assert_promise_would_raise_exception (playlist_of_id test_playlist_id expired_token)
      ("request metadata for playlist (id = " ^ test_playlist_id ^ ") failed: 401 The access token expired") in 
  Lwt.return (); 
;;

(* exceptions from using a good token, but bad id *)
let test_bad_id_exceptions _ = 
  let%lwt token = get_new_api_token () in
  (* ill-formed ids *)
  let%lwt () = assert_promise_would_raise_exception (song_of_id ill_formed_song_id token)
      ("request features for song (id = " ^ ill_formed_song_id ^ ") failed: 400 invalid request") in 
  let%lwt () = assert_promise_would_raise_exception (playlist_of_id ill_formed_playlist_id token)
      ("request metadata for playlist (id = " ^ ill_formed_playlist_id ^ ") failed: 404 Invalid playlist Id") in 
  (* requesting a song with the id of a playlist and vice versa *)
  let%lwt () = assert_promise_would_raise_exception (song_of_id test_playlist_id token)
      ("request features for song (id = " ^ test_playlist_id ^ ") failed: 404 analysis not found") in 
  let%lwt () = assert_promise_would_raise_exception (playlist_of_id test_song_id_skylines token)
      ("request metadata for playlist (id = " ^ test_song_id_skylines ^ ") failed: 404 Not found.") in 
  Lwt.return (); 
;;


let spotify_tests =
  "Spotify" >: test_list [
    "Get Access Token" >:: lwt_wrapper test_get_token;
    "Get Song" >:: lwt_wrapper test_get_song;
    "Get Playlist" >:: lwt_wrapper test_get_playlist;
    "Exceptions Due to Bad Token" >:: lwt_wrapper test_bad_token_exceptions; 
    "Exceptions Due to Bad Song/Playlist Id" >:: lwt_wrapper test_bad_id_exceptions;
  ]

let series =
  "Final Project Tests" >::: [
    numpy_tests;
    svm_tests;
    machine_learning_tests;
    spotify_tests; 
  ]

let () = 
  run_test_tt_main series