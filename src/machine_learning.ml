module Np = Np.Numpy

open Core
open Spotify

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
  val equal: t -> t -> bool
end 

module type Classification = sig
  type t
  type dataset
  val randomize : playlist -> playlist
  val balance_classes : playlist -> playlist -> (playlist * playlist)
  val normalize : playlist -> playlist -> (playlist * playlist)
  val standardize : playlist -> playlist -> (playlist * playlist)
  val split : playlist -> playlist -> float -> float -> dataset
  (*val save_dataset : dataset -> string -> unit
  val load_dataset : string -> dataset*)
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
  type dataset = {pos_train: playlist; pos_valid: playlist; pos_test: playlist; neg_train: playlist; neg_valid: playlist; neg_test: playlist}
  (* classify a song represented by a vector into one of the two playlists true = first, false = second *)
  let classify (c: t) (s: song) : string =
    match Classifier.classes c, s with 
    | (class1, class2), {features_vector; _} 
      -> if Classifier.predict c features_vector then class1 else class2

  let rows (matrix: Np.Ndarray.t) (first: int) (last: int) : Np.Ndarray.t =
    Np.Ndarray.get ~key:[Np.slice ~i:first ~j:last (); 
                         Np.slice ~i:0 ~j:(Np.size ~axis:1 matrix) ()] matrix

  let column (matrix: Np.Ndarray.t) (index: int) : Np.Ndarray.t =
    Np.Ndarray.get ~key:[Np.slice ~i:0 ~j:(Np.size ~axis:1 matrix) (); 
                         Np.slice ~i:index ~j:(index + 1) ()] matrix

  let map_vector (vector: Np.Ndarray.t) (f: float -> float) : Np.Ndarray.t = 
    Np.Ndarray.of_float_array @@ Array.map ~f @@ Np.Ndarray.to_float_array vector 

  let replace_features (p: playlist) (f: Np.Ndarray.t) : playlist =
    {features_matrix = f; name = p.name; pid = p.pid}

    let split (pos_playlist: playlist) (neg_playlist: playlist) (valid: float) (test: float) : dataset =
      let pos_valid_bound = Np.size ~axis:0 pos_playlist.features_matrix |> Float.of_int |> Float.( * ) valid |> Float.round_nearest |> Int.of_float
      in let pos_test_bound = Np.size ~axis:0 pos_playlist.features_matrix |> Float.of_int |> Float.( * ) (Float.(+) valid test) |> Float.round_nearest |> Int.of_float
    in let pos_train_bound = Np.size ~axis:0 pos_playlist.features_matrix
    in let neg_valid_bound = Np.size ~axis:0 neg_playlist.features_matrix |> Float.of_int |> Float.( * ) valid |> Float.round_nearest |> Int.of_float
    in let neg_test_bound = Np.size ~axis:0 neg_playlist.features_matrix |> Float.of_int |> Float.( * ) (Float.(+) valid test) |> Float.round_nearest |> Int.of_float
  in let neg_train_bound = Np.size ~axis:0 neg_playlist.features_matrix
  
      in if 0 >= pos_valid_bound || 0 >= neg_valid_bound then failwith "Validation split size too small" 
      else if pos_valid_bound >= pos_test_bound || neg_valid_bound >= neg_test_bound then failwith "Test split size too small" 
      else if pos_test_bound >= pos_train_bound || neg_test_bound >= neg_train_bound then failwith "Train split size too small (validation + test too big)" 
      else
      {pos_valid = replace_features pos_playlist @@ rows pos_playlist.features_matrix 0 pos_valid_bound;
       pos_test = replace_features pos_playlist @@ rows pos_playlist.features_matrix pos_valid_bound pos_test_bound;
       pos_train = replace_features pos_playlist @@ rows pos_playlist.features_matrix pos_test_bound pos_train_bound;
       neg_valid = replace_features neg_playlist @@ rows neg_playlist.features_matrix 0 neg_valid_bound;
       neg_test = replace_features neg_playlist @@ rows neg_playlist.features_matrix neg_valid_bound neg_test_bound;
       neg_train = replace_features neg_playlist @@rows neg_playlist.features_matrix neg_test_bound neg_train_bound;}
   
  let randomize (p: playlist) : playlist =
    replace_features p @@ Np.Ndarray.of_pyobject @@ Np.Random.shuffle p.features_matrix

  let balance_classes (pos_playlist: playlist) (neg_playlist: playlist) : (playlist * playlist) =
    let len = Int.min (Np.size ~axis:1 pos_playlist.features_matrix) 
        (Np.size ~axis:1 neg_playlist.features_matrix)
    in (replace_features pos_playlist @@ Np.Ndarray.get 
          ~key:[Np.slice ~i:0 ~j:len (); 
                Np.slice ~i:0 ~j:(Np.size ~axis:1 pos_playlist.features_matrix) ()] 
          pos_playlist.features_matrix,
        replace_features neg_playlist @@ Np.Ndarray.get 
          ~key:[Np.slice ~i:0 ~j:len (); 
                Np.slice ~i:0 ~j:(Np.size ~axis:1 neg_playlist.features_matrix) ()] 
          neg_playlist.features_matrix)

  let normalize (pos_playlist: playlist) (neg_playlist: playlist) : (playlist * playlist) =
    let all_data = Np.append ~axis:0 ~arr:pos_playlist.features_matrix 
        ~values:neg_playlist.features_matrix ()
    in let min_max = Array.zip_exn (Np.Ndarray.to_float_array @@ Np.min ~axis:[1] all_data)
           (Np.Ndarray.to_float_array @@ Np.max ~axis:[1] all_data)
    in (replace_features pos_playlist 
          (Array.foldi min_max ~init:(Np.empty [Np.size ~axis:0 pos_playlist.features_matrix; 0]) 
             ~f:(fun col arr (min, max) -> Np.append ~axis:1 ~arr 
                    ~values:(map_vector (column pos_playlist.features_matrix col) 
                               (fun old -> Float.(/) (Float.(-) old min)(Float.(-) max min))) ())), 
        replace_features neg_playlist 
          (Array.foldi min_max ~init:(Np.empty [Np.size ~axis:0 neg_playlist.features_matrix; 0]) 
             ~f:(fun col arr (min, max) -> Np.append ~axis:1 ~arr 
                    ~values:(map_vector (column neg_playlist.features_matrix col) 
                               (fun old -> Float.(/) (Float.(-) old min)(Float.(-) max min))) ())))

  let standardize (pos_playlist: playlist) (neg_playlist: playlist) : (playlist * playlist) =
    let all_data = Np.append ~axis:0 ~arr:pos_playlist.features_matrix 
        ~values:neg_playlist.features_matrix ()
    in let mean_std = Array.zip_exn (Np.Ndarray.to_float_array @@ Np.mean ~axis:[1] all_data)
           (Np.Ndarray.to_float_array @@ Np.std ~axis:[1] all_data)
    in (replace_features pos_playlist 
          (Array.foldi mean_std ~init:(Np.empty [Np.size ~axis:0 pos_playlist.features_matrix; 0]) 
             ~f:(fun col arr (mean, std) -> Np.append ~axis:1 ~arr 
                    ~values:(map_vector (column pos_playlist.features_matrix col) 
                               (fun old -> Float.(/) (Float.(-) old mean)std)) ())), 
        replace_features neg_playlist 
          (Array.foldi mean_std ~init:(Np.empty [Np.size ~axis:0 neg_playlist.features_matrix; 0]) 
             ~f:(fun col arr (mean, std) -> Np.append ~axis:1 ~arr 
                    ~values:(map_vector (column neg_playlist.features_matrix col) 
                               (fun old -> Float.(/) (Float.(-) old mean)std)) ())))  

  let rec test_sample_i (c: t) (samples: Np.Ndarray.t) (pos: int) (index: int) : int =
    if (index >= Np.size ~axis:0 samples) then pos 
    else (test_sample_i c samples 
            (if Classifier.predict c @@ rows samples index (index + 1) then pos + 1 else pos) (index + 1))

  let test (c: t) (pos: playlist) (neg: playlist) : confusion_matrix =
    match pos, neg with 
      {features_matrix = posFeatures; _}, {features_matrix = negFeatures; _} -> 
      let tp = test_sample_i c posFeatures 0 0
      in let fp = test_sample_i c negFeatures 0 0
      in let fn = Np.size ~axis:0 posFeatures - tp
      in let tn = Np.size ~axis:0 negFeatures - fp
      in {tp; fp; fn; tn}

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