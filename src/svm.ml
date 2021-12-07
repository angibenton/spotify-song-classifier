module Np = Np.Numpy

open Sklearn.Svm
open Core
open Machine_learning
open Spotify

type svm = {hyperplane: Np.Ndarray.t; intercept: float; class1: string; class2: string}


module SVM_Model = struct
  (* the model *)
  type t = svm

  type hyperparameters = float

  (* save a model into a file with a give filename *)
  let save (svc: t) (file: string) : unit = 
    let f = Stdio.Out_channel.create file
    in match svc with |
      {hyperplane; intercept; class1; class2} 
      -> Stdio.Out_channel.output_string f (class1 ^ "\n" ^ class2 ^ "\n" ^ Float.to_string intercept ^ "\n" ^
                                            (Array.fold (Np.Ndarray.to_float_array hyperplane) 
                                               ~init:"" ~f:(fun s v -> s ^ " " ^ Float.to_string v)));
      Stdio.Out_channel.flush f;
      Stdio.Out_channel.close f


  (* open up a model file with a given filename, parse a model object from it *)
  let load (file: string) : t =
    match Stdio.In_channel.read_lines file with 
    | class1 :: class2 :: intercept ::arr :: [] -> String.split ~on:' ' arr 
                                       |> List.filter ~f:(fun (s) -> not @@ String.is_empty s)
                                       |> List.map ~f:Float.of_string |> Np.Ndarray.of_float_list 
                                       |> fun (hyperplane) -> {hyperplane; intercept = Float.of_string intercept; class1; class2;}
    | _ -> failwith "improper file formatting"

  (* train a binary classifier on two playlists represented as tensors *)
  let train (c: hyperparameters) (p1: playlist) (p2: playlist) : t =
    let x, y = match p1, p2 with
      | {features_matrix = features1; _}, {features_matrix = features2; _} 
        -> Array.init (Np.size ~axis:0 features1) ~f:(fun _ -> -1) |> fun (arr1) 
           -> (Np.append ~axis:0 ~arr:features1 () ~values:features2), 
              (Array.append arr1 @@ Array.init (Np.size ~axis:0 features2) ~f:(fun _ -> 1) 
               |> Np.Ndarray.of_int_array)
    in let clf = LinearSVC.create ~c ~dual:false () 
    in match p1, p2 with 
    | {name = class1; _}, {name = class2; _} 
      -> LinearSVC.fit clf ~x ~y |> fun svc -> {hyperplane = (LinearSVC.coef_ svc); 
  intercept=(Array.get (Np.Ndarray.to_float_array @@ LinearSVC.intercept_ svc) 0); class1; class2}

  let predict_score (hyperplane: Np.Ndarray.t) (intercept: float) (features: Np.Ndarray.t) : float =
    let normalize = Np.dot hyperplane ~b:hyperplane |> Np.Ndarray.to_float_array
                                     |> fun (arr) -> Array.get arr 0 |> Float.sqrt
    in Float.(/) (Float.(+) intercept @@ Array.get (Np.Ndarray.to_float_array @@ 
                             Np.dot ~b:features (Np.reshape ~newshape:[(Np.Ndarray.size hyperplane); 1] hyperplane)) 0) normalize

  let predict (svc: t) (features: Np.Ndarray.t) : bool =
    match svc with 
    | {hyperplane; intercept; _} ->Float.(>) 0. @@ predict_score hyperplane intercept features

  let classes (svc: t) : string * string = 
    match svc with
    | {class1; class2; _} -> (class1, class2)
end 

module SVM_Classification = Classification(SVM_Model)