module Np = Np.Numpy

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
  (* give the class names of the model *)
  val classes: t -> string * string
end 

module type Classification = sig
  type t
  (* classify a song represented by a vector into one of the two playlists *)
  val classify : t -> song -> string
  (* return the confusion matrix from testing the model on a tensor of labeled songs *)
  val test : t -> playlist -> playlist -> confusion_matrix
  (* calculate the accuracy of a test result confusion matrix *)
  val accuracy : confusion_matrix -> float
  (* calculate the F1 Score of a test result confusion matrix *)
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