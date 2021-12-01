module Np = Np.Numpy

open Sklearn.Svm
open Core
open Machine_learning

module SVM_Model = struct
  (* the model *)
  type t = svm

  (* save a model into a file with a give filename *)
  let save (svc: t) (file: string) : unit = 
    let f = Stdio.Out_channel.create file
    in match svc with |
      {hyperplane; class1; class2} 
      -> Stdio.Out_channel.output_string f (class1 ^ "\n" ^ class2 ^ "\n" ^ 
                                            (Array.fold (Np.Ndarray.to_float_array hyperplane) 
                                               ~init:"" ~f:(fun s v -> s ^ " " ^ Float.to_string v)));
      Stdio.Out_channel.flush f;
      Stdio.Out_channel.close f


  (* open up a model file with a given filename, parse a model object from it *)
  let load (file: string) : t =
    match Stdio.In_channel.read_lines file with 
    | class1 :: class2 :: arr :: [] -> String.split ~on:' ' arr 
                                       |> List.filter ~f:(fun (s) -> not @@ String.is_empty s)
                                       |> List.map ~f:Float.of_string |> Np.Ndarray.of_float_list 
                                       |> fun (hyperplane) -> {hyperplane; class1; class2;}
    | some -> List.iter some ~f:(Printf.printf "%s"); {hyperplane=(Np.Ndarray.of_int_list [2]); class1 ="l"; class2="f"}

  (* train a binary classifier on two playlists represented as tensors *)
  let train (p1: playlist) (p2: playlist) : t =
    let x, y = match p1, p2 with
      | {features = features1; _}, {features = features2; _} 
        -> Array.init (Np.size ~axis:0 features1) ~f:(fun _ -> -1) |> fun (arr1) 
           -> (Np.append ~axis:0 ~arr:features1 () ~values:features2), 
              (Array.append arr1 @@ Array.init (Np.size ~axis:0 features2) ~f:(fun _ -> 1) 
               |> Np.Ndarray.of_int_array)
    in let clf = LinearSVC.create ~c:1.0 ~dual:false () 
    in match p1, p2 with 
    | {name = class1; _}, {name = class2; _} 
      -> {hyperplane = (LinearSVC.fit clf ~x ~y |> fun (svc) 
                        -> Np.append ~arr:(LinearSVC.coef_ svc) () 
                          ~values:(LinearSVC.intercept_ svc)); class1; class2}

  let predict_score (hyperplane: Np.Ndarray.t) (features: Np.Ndarray.t) : float =
    let normalize = Np.Ndarray.get ~key:[Np.slice ~i:0 ~j:12 ();] hyperplane 
                    |> fun (coef) -> Np.dot coef ~b:coef |> Np.Ndarray.to_float_array
                                     |> fun (arr) -> Array.get arr 0 |> Float.sqrt
    in Float.(/) (Array.get (Np.Ndarray.to_float_array @@ 
                             Np.dot ~b:(Np.append ~arr:features () 
                                          ~values:(Np.vectori [|1|])) hyperplane) 0) normalize

  let predict (svc: t) (features: Np.Ndarray.t) : bool =
    match svc with 
    | {hyperplane; _} ->Float.(>) 0. @@ predict_score hyperplane features
  
  let classes (svc: t) : string * string = 
    match svc with
    | {class1; class2; _} -> (class1, class2)
end 

module SVM_Classification = Classification(SVM_Model)

(*let () = 
  let playlist1 = {name = "the first one"; id = "123"; 
                   features = (Np.matrixf [| [| -1.; -1. |]; [| -2.; -1. |]; [| 4.; 3.2|]; [| 2.; 1. |] |])}
  in let playlist2 = {name = "the second one"; id = "124"; 
                      features = (Np.matrixf [| [| -1.; -1. |];  |])}
  in let classifier = SVM_Model.train playlist1 playlist2
  in SVM_Model.save classifier "TestSVM.txt";
  let perf = SVM_Classification.test classifier playlist1 playlist2 
  in Printf.printf "Training accuracy: %f" @@ SVM_Classification.accuracy perf;
  Printf.printf "Training F1 Score: %f" @@ SVM_Classification.f1_score perf;
  SVM_Model.load "TestSVM.txt" |> fun (svc) -> Printf.printf "%s"
  @@ SVM_Classification.classify svc {features = Np.vectorf [| 2.; 1. |]; name = "hi"; id = "h"}*)
