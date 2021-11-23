module Np = Np.Numpy

open Sklearn.Svm
open Core

type song = {name: string; id: string; features: Np.Ndarray.t;}

type playlist = {name: string; id: string; features: Np.Ndarray.t;}

type svm = {hyperplane: Np.Ndarray.t; class1: string; class2: string}

module Model = struct
  (* the model *)
  type t = svm

  (* save a model into a file with a give filename *)
  let save (svc: t) (file: string) : unit = 
    match svc with |
      {hyperplane; class1; class2} 
      -> Stdio.Out_channel.create file |> fun (f) 
         -> Stdio.Out_channel.output_string f (class1 ^ "\n" ^ class2 ^ "\n" ^ 
                                               (Array.fold (Np.Ndarray.to_float_array hyperplane) 
                                                  ~init:"" ~f:(fun s v -> s ^ " " ^ Float.to_string v)))

  (* open up a model file with a given filename, parse a model object from it *)
  let load (file: string) : t =
    List.iter (Stdio.In_channel.read_lines file) ~f:(Printf.printf "%s");
    match Stdio.In_channel.read_lines file with 
    | class1 :: class2 :: arr :: [] -> String.split ~on:' ' arr 
                                       |> List.map ~f:Float.of_string |> Np.Ndarray.of_float_list 
                                       |> fun (hyperplane) -> {hyperplane; class1; class2;}
    | some -> List.iter some ~f:(Printf.printf "%s"); {hyperplane=(Np.Ndarray.of_int_list [2]); class1 ="l"; class2="f"}

  (* train a binary classifier on two playlists represented as tensors *)
  let train (p1: playlist) (p2: playlist) : t =
    let x, y = match p1, p2 with
      | {features = features1; _}, {features = features2; _} 
        -> Array.init (Np.size features1) ~f:(fun _ -> -1) |> fun (arr1) 
           -> (Np.append ~arr:features1 () ~values:features2), 
              Array.append arr1 @@ Array.init (Np.size features2) ~f:(fun _ -> 1) |> Np.Ndarray.of_int_array 
    in let clf = LinearSVC.create ~c:1.0 () 
    in match p1, p2 with 
    | {name = class1; _}, {name = class2; _} 
      -> {hyperplane = (LinearSVC.fit clf ~x ~y |> fun (svm) 
                        -> Np.append ~arr:(LinearSVC.coef_ svm) () ~values:(LinearSVC.intercept_ svm)); class1; class2}

  (* classify a song represented by a vector into one of the two playlists true = first, false = second *)
  let classify (svc: t) (s: song) : string =
    match svc, s with 
    | {hyperplane; class1; class2}, {features; _} -> 
      if 0 < Array.get (Np.Ndarray.to_int_array @@ Np.dot ~b:hyperplane features) 0 then class1 else class2

end 

let () = 
  let y = Np.vectori [| 1; 1; 2; 2 |] in
  let x = Np.matrixf [| [| -1.; -1. |]; [| -2.; -1. |]; [| -1.; -1. |]; [| 2.; 1. |] |]
  in
  let clf = LinearSVC.create ~c:1.0 () in
  Format.printf "%a\n" LinearSVC.pp @@ LinearSVC.fit clf ~x ~y;
  Format.printf "%a\n" Np.pp (LinearSVC.coef_ clf);
  Format.printf "%a\n" Np.pp (LinearSVC.intercept_ clf);
  Model.save {hyperplane = (Np.append ~arr:(LinearSVC.coef_ clf) () ~values:(LinearSVC.intercept_ clf)); class1 = "1"; class2 = "2"} "TestSVM.txt";
  Model.load "TestSVM.txt" |> fun (_) -> ();
