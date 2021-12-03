module Np = Np.Numpy

open Core

type song = {name: string; id: string; features: Np.Ndarray.t;}

type playlist = {name: string; id: string; features: Np.Ndarray.t;}

type confusion_matrix = {tp: int; fp: int; tn: int; fn: int}

module type Model = sig 
  (* the model *)
  type t
  type hyperparameters

  (* save a model into a file with a give filename *)
  val save : t -> string -> unit
  (* open up a model file with a given filename, parse a model object from it *)
  val load : string -> t
  (* train a binary classifier on two playlists represented as tensors *)
  val train : hyperparameters -> playlist -> playlist -> t
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
  (* format the confusion matrix to be printed *)
  val pretty_confusion : confusion_matrix -> string
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
      in let fn = Np.size ~axis:0 posFeatures - tp
      in let tn = Np.size ~axis:0 negFeatures - fp
      in Printf.printf "%d%d%d%d" tp fp fn tn; {tp; fp; fn; tn}

  let pretty_confusion (cm: confusion_matrix) : string =

    let (tp, fp, tn, fn) = match cm with 
      | {tp; fp; tn; fn} -> (tp, fp, tn, fn)

    in let padding = (fun(_) -> ' ')
    in let left_margin = String.init 10 ~f:padding

    in let margin = left_margin ^ String.init 4 ~f:padding

    in let digits (n: int) : int =
        match n with
        | 0 -> 1
        | _ -> Float.of_int n |> Float.log10 |> fun(res) -> Int.of_float res + 1

    in let max_width = List.fold_left (tp :: fp :: tn :: fn :: [])
           ~f:(fun max curr -> if curr > max then curr else max) 
           ~init:Int.min_value |> digits

    in let spaced_int (n: int) : string =
         String.init (max_width - digits n) ~f:padding ^ Int.to_string n

    in let total_width = max_width * 2 + 7 

    in let line = String.init total_width ~f:(fun (_) -> '-') ^ "\n"

    in let row_1 = 
         let spacing = String.init ((total_width - 6) / 2)  ~f:padding
         in margin ^ spacing ^ "actual " ^ spacing ^ "\n"

    in let row_2 = 
         let size = (total_width - 6) / 3
         in let base_spacing = String.init size ~f:padding
         in let extra_spacing = String.init (size + 1) ~f:padding
         in margin ^ (match (total_width - 6) % 3 with 
             | 0 -> base_spacing ^ "pos" ^ base_spacing ^ "neg" ^ base_spacing
             | 1 -> base_spacing ^ "pos" ^ extra_spacing ^ "neg" ^ base_spacing
             | 2 -> extra_spacing ^ "pos" ^ base_spacing ^ "neg" ^ extra_spacing
             | _ -> assert false) ^ "\n"

    in let row_3 = margin ^ line
    in let row_4 = left_margin ^ "pos" ^ " | " ^ (spaced_int tp) ^ " | " ^ (spaced_int fp) ^ " |\n" 

    in let row_5 = "predicted     " ^ line
    in let row_6 = left_margin ^ "neg" ^ " | " ^ (spaced_int fn) ^ " | " ^ (spaced_int tn) ^ " |\n" 
    in let row_7 = margin ^ line
    in row_1 ^ row_2 ^ row_3 ^ row_4 ^ row_5 ^ row_6 ^ row_7


  let accuracy (cm: confusion_matrix) : float =
    match cm with 
    | {tp; tn; fp; fn;} -> Float.(/) (Int.to_float @@ tp + tn) (Int.to_float @@ tp + tn + fp + fn) 

  let f1_score (cm: confusion_matrix) : float =
    match cm with 
    | {tp; fp; fn; _} -> Float.(/) (Int.to_float tp) 
                           (Float.(+) (Int.to_float tp) @@ Float.( * ) 0.5 @@ Int.to_float @@ fp + fn) 
end