module Np = Np.Numpy

open Sklearn.Svm
open Core

type song = {name: string; id: string; features: Np.Ndarray.t;}

type playlist = {name: string; id: string; features: Np.Ndarray.t;}

type svm = {hyperplane: Np.Ndarray.t; class1: string; class2: string}

type confusion_matrix = {tp: int; fp: int; tn: int; fn: int}

module type Model = sig 
  (* the model *)
  type t 

  (* save a model into a file with a give filename *)
  val save : t -> string -> unit
  (* open up a model file with a given filename, parse a model object from it *)
  val load : string -> t
  (* train a binary classifier on two playlists represented as tensors *)
  val train : playlist -> playlist -> t
  (* predict the class a new feature vector belongs to, true being positive class *)
  val predict : t -> Np.Ndarray.t -> bool
  val classes: t -> string * string
end 

module type Classification = sig
  type t
  (* classify a song represented by a vector into one of the two playlists *)
  val classify : t -> song -> string
  (* return the confusion matrix from testing the model on a tensor of labeled songs *)
  val test : t -> playlist -> playlist -> confusion_matrix
  val accuracy : confusion_matrix -> float
  val f1_score : confusion_matrix -> float
end

module Classification (Classifier: Model) : (Classification with type t = Classifier.t) = struct
  type t = Classifier.t

  (* classify a song represented by a vector into one of the two playlists true = first, false = second *)
  let classify (c: t) (s: song) : string =
    match Classifier.classes c, s with 
    | (class1, class2), {features; _} 
      -> if Classifier.predict c features then class1 else class2

  let row (matrix: Np.Ndarray.t) (index: int) : Np.Ndarray.t =
    Np.Ndarray.get ~key:[Np.slice ~i:index ~j:(index + 1) (); Np.slice ~i:0 ~j:(Np.size ~axis:1 matrix) ()] matrix

  let rec test_sample_i (c: t) (samples: Np.Ndarray.t) (pos: int) (index: int) : int =
    if (index >= Np.size ~axis:0 samples) then pos 
    else (test_sample_i c samples (if Classifier.predict c @@ row samples index then pos + 1 else pos) (index + 1))

  let test (c: t) (pos: playlist) (neg: playlist) : confusion_matrix =
    match pos, neg with 
      {features = posFeatures; _}, {features = negFeatures; _} -> 
      let tp = test_sample_i c posFeatures 0 0
      in let fp = test_sample_i c negFeatures 0 0
      in let fn = Np.size ~axis:0 posFeatures
      in let tn = Np.size ~axis:0 negFeatures
      in {tp; fp; fn; tn}

  let accuracy (cm: confusion_matrix) : float =
    match cm with 
    | {tp; tn; fp; fn;} -> Float.(/) (Int.to_float @@ tp + tn) (Int.to_float @@ tp + tn + fp + fn) 

  let f1_score (cm: confusion_matrix) : float =
    match cm with 
    | {tp; fp; fn; _} -> Float.(/) (Int.to_float tp) 
                           (Float.(+) (Int.to_float tp) @@ Float.( * ) 0.5 @@ Int.to_float @@ fp + fn) 
end

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
                                     |> fun (arr) -> Array.get arr 0
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

let () = 
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
  @@ SVM_Classification.classify svc {features = Np.vectorf [| 2.; 1. |]; name = "hi"; id = "h"}
