open Numpy_helper
open Core
open Spotify

type confusion_matrix = {tp: int; fp: int; tn: int; fn: int}
type dataset = {pos_train: playlist; pos_valid: playlist; 
pos_test: playlist; neg_train: playlist; 
neg_valid: playlist; neg_test: playlist; 
shift: Np.Ndarray.t; scale: Np.Ndarray.t}

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
  (* train many classifiers with different hyperparameter combinations *)
  val tune : hyperparameters list -> playlist -> playlist -> t list
  (* predict the class a new feature vector belongs to, true being positive class *)
  val predict : t -> Np.Ndarray.t -> bool
  (* give the class names of the model *)
  val classes: t -> string * string
  val equal: t -> t -> bool
  val preprocess_features: t -> (float * float) list
end 

module type Classification = sig
  type t
  val randomize : playlist -> playlist
  val balance_classes : (playlist * playlist) -> (playlist * playlist)
  val normalize : (playlist * playlist) -> (playlist * playlist * ((float * float) list))
  val standardize : (playlist * playlist) -> (playlist * playlist * ((float * float) list))
  val split : (playlist * playlist) -> float -> float -> (float * float) list -> dataset
  val save_dataset : dataset -> string -> unit
  val load_dataset : string -> dataset
  (* classify a song represented by a vector into one of the two playlists *)
  val classify : t -> song -> string
  (* return the confusion matrix from testing the model on a tensor of labeled songs *)
  val test : t -> playlist -> playlist -> confusion_matrix
  (* choose the best model on the validation set based on some evaulation function *)
  val tune : t list -> playlist -> playlist -> (confusion_matrix -> float) -> t
  (* format the confusion matrix to be printed *)
  val pretty_confusion : confusion_matrix -> string
  (* calculate the accuracy of a test result confusion matrix *)
  val accuracy : confusion_matrix -> float
  (* calculate the F1 Score of a test result confusion matrix *)
  val f1_score : confusion_matrix -> float

end

module Classification (Classifier: Model) : (Classification with type t = Classifier.t) = struct
  type t = Classifier.t
  let save_dataset (d: dataset) (folder: string) : unit =
    Unix.mkdir folder; save_playlist d.pos_train @@ folder ^ "/pos_train";
    save_playlist d.pos_valid @@ folder ^ "/pos_val";
    save_playlist d.pos_test @@ folder ^ "/pos_test";
    save_playlist d.neg_train @@ folder ^ "/neg_train";
    save_playlist d.neg_valid @@ folder ^ "/neg_val";
    save_playlist d.neg_test @@ folder ^ "/neg_test";
    let f = Stdio.Out_channel.create @@ folder ^ "/preprocess"
    in Stdio.Out_channel.output_string f @@ (vector_to_string d.shift |> fun s -> if String.is_empty s then "none" else s) ^ "\n" ^ 
    (vector_to_string d.scale |> fun s -> if String.is_empty s then "none" else s);
    Stdio.Out_channel.flush f;
    Stdio.Out_channel.close f
  let load_dataset (folder: string) : dataset =
    (match Stdio.In_channel.read_lines @@ folder ^ "/preprocess" with 
    | shift :: scale :: _ -> {pos_train = load_playlist @@ folder ^ "/pos_train";
    pos_valid = load_playlist @@ folder ^ "/pos_val";
    pos_test = load_playlist @@ folder ^ "/pos_test";
    neg_train = load_playlist @@ folder ^ "/neg_train";
    neg_valid = load_playlist @@ folder ^ "/neg_val";
    neg_test = load_playlist @@ folder ^ "/neg_test"; 
    shift = txt_to_vec shift; scale = txt_to_vec scale}
    | _ -> failwith "improper file formatting")
    

  let split ((pos, neg): (playlist * playlist)) (valid: float) (test: float) (preprocess: (float * float) list) : dataset =
    let pos_valid_bound = Np.size ~axis:0 pos.features_matrix 
                          |> Float.of_int |> Float.( * ) valid 
                          |> Float.round_nearest |> Int.of_float
    in let pos_test_bound = Np.size ~axis:0 pos.features_matrix 
                            |> Float.of_int |> Float.( * ) (Float.(+) valid test) 
                            |> Float.round_nearest |> Int.of_float
    in let pos_train_bound = Np.size ~axis:0 pos.features_matrix
    in let neg_valid_bound = Np.size ~axis:0 neg.features_matrix 
                             |> Float.of_int |> Float.( * ) valid 
                             |> Float.round_nearest |> Int.of_float
    in let neg_test_bound = Np.size ~axis:0 neg.features_matrix 
                            |> Float.of_int |> Float.( * ) (Float.(+) valid test) 
                            |> Float.round_nearest |> Int.of_float
    in let neg_train_bound = Np.size ~axis:0 neg.features_matrix

    in if 0 >= pos_valid_bound || 0 >= neg_valid_bound 
    then failwith "Validation split size too small" 
    else if pos_valid_bound >= pos_test_bound || neg_valid_bound >= neg_test_bound 
    then failwith "Test split size too small" 
    else if pos_test_bound >= pos_train_bound || neg_test_bound >= neg_train_bound 
    then failwith "Train split size too small (validation + test too big)" 
    else List.unzip preprocess |> fun (shift, scale) -> 
      {pos_valid = replace_features pos @@ rows pos.features_matrix 0 pos_valid_bound;
       pos_test = replace_features pos @@ rows pos.features_matrix pos_valid_bound pos_test_bound;
       pos_train = replace_features pos @@ rows pos.features_matrix pos_test_bound pos_train_bound;
       neg_valid = replace_features neg @@ rows neg.features_matrix 0 neg_valid_bound;
       neg_test = replace_features neg @@ rows neg.features_matrix neg_valid_bound neg_test_bound;
       neg_train = replace_features neg @@rows neg.features_matrix neg_test_bound neg_train_bound;
       shift = Np.Ndarray.of_float_list shift; scale = Np.Ndarray.of_float_list scale}

  let randomize (p: playlist) : playlist =
    List.init (Np.size ~axis:0 p.features_matrix) 
      ~f: (fun index -> rows p.features_matrix index (index + 1) 
                        |> Np.Ndarray.to_float_array)
    |> fun l -> List.iter l 
      ~f:(fun row -> Array.sort row ~compare:(fun _ _ -> (Random.int 3) - 1)); 
    List.map l ~f:(fun row 
                    -> Np.Ndarray.of_float_array row 
                       |> Np.reshape ~newshape:[1; ((Np.size ~axis:1 p.features_matrix))]) 
    |> fun l -> replace_features p @@ matrix_of_vector_list l

  let balance_classes ((pos, neg): (playlist * playlist)) : (playlist * playlist) =
    let len = Int.max (Np.size ~axis:0 pos.features_matrix) 
        (Np.size ~axis:0 neg.features_matrix)
    in (replace_features pos @@ rows pos.features_matrix 0 len,
        replace_features neg @@ rows neg.features_matrix 0 len)

  let clean_features (features: Np.Ndarray.t) (stat: (float * float) list) : Np.Ndarray.t =
    let f (col: int) (arr: Np.Ndarray.t) ((shift, scale): (float * float)) : Np.Ndarray.t = Np.append ~axis:1 ~arr 
        ~values:((map_vector (columns features col (col + 1)) 
                    (fun elem -> Float.(/) (Float.(-) elem shift) scale)) 
                 |> fun v -> Np.reshape ~newshape:[Np.size ~axis:0 features; 1] v) ()
    in List.foldi stat ~init:(Np.empty [Np.size ~axis:0 features; 0]) ~f 

  let arr_min (arr: float array) : float =
    Array.fold arr ~init:Float.infinity 
      ~f:(fun min elem -> if Float.(<) elem min then elem else min)

  let arr_max (arr: float array) : float =
    Array.fold arr ~init:Float.neg_infinity 
      ~f:(fun max elem -> if Float.(<) max elem then elem else max)

  let arr_mean (arr: float array) : float =
    Array.fold arr ~init:(0., 0.) 
      ~f:(fun (sum, count) elem -> (Float.(+) sum elem, Float.(+) count 1.)) 
    |> fun (sum, count) -> Float.(/) sum count

  let arr_std (arr: float array) (mean: float): float =
    Array.fold arr ~init:(0., 0.) 
      ~f:(fun (sum, count) elem 
           -> (Float.(+) sum @@ Float.square @@ Float.(-) elem mean, Float.(+) count 1.)) 
    |> fun (sum, count) -> Float.sqrt @@ Float.(/) sum count

  let normalize ((pos, neg): (playlist * playlist)) : (playlist * playlist * ((float * float) list)) =
    let cols = List.init (Np.size ~axis:1 pos.features_matrix) 
        ~f:(fun col -> columns (Np.append ~axis:0 ~arr:pos.features_matrix 
                                  ~values:neg.features_matrix ()) col (col + 1) 
                       |> Np.Ndarray.to_float_array)
    in let min_max = 
         List.map cols ~f:(fun arr -> (arr_min arr, Float.(-) (arr_min arr) (arr_max arr)))
    in (replace_features pos @@ clean_features pos.features_matrix min_max, 
        replace_features neg @@ clean_features neg.features_matrix min_max, min_max)

  let standardize ((pos, neg): (playlist * playlist)) : (playlist * playlist * ((float * float) list)) =
    let cols = List.init (Np.size ~axis:1 pos.features_matrix) 
        ~f:(fun col -> columns (Np.append ~axis:0 ~arr:pos.features_matrix 
                                  ~values:neg.features_matrix ()) col (col + 1) 
                       |> Np.Ndarray.to_float_array)
    in let mean_std = 
         List.map cols ~f:(fun arr -> (arr_mean arr, arr_std arr @@ arr_mean arr))
    in (replace_features pos @@ clean_features pos.features_matrix mean_std, 
        replace_features neg @@ clean_features neg.features_matrix mean_std, mean_std)  

  (* classify a song represented by a vector into one of the two playlists 
     true = first, false = second *)
  let classify (c: t) (s: song) : string =
    (if 0 = List.length @@ Classifier.preprocess_features c then s else {features_vector = clean_features s.features_vector @@ Classifier.preprocess_features c; name = s.name; sid = s.sid}) |> fun cleaned -> 
    match Classifier.classes c, cleaned with 
    | (class1, class2), {features_vector; _} 
      -> if Classifier.predict c features_vector then class1 else class2

  let rec test_sample_i (c: t) (samples: Np.Ndarray.t) (pos: int) (index: int) : int =
    if (index >= Np.size ~axis:0 samples) then pos 
    else (test_sample_i c samples 
            (if Classifier.predict c @@ rows samples index (index + 1) 
             then pos + 1 else pos) (index + 1))

  let test (c: t) (pos: playlist) (neg: playlist) : confusion_matrix =
    match pos, neg with 
      {features_matrix = posFeatures; _}, {features_matrix = negFeatures; _} -> 
      let tp = test_sample_i c posFeatures 0 0
      in let fp = test_sample_i c negFeatures 0 0
      in let fn = Np.size ~axis:0 posFeatures - tp
      in let tn = Np.size ~axis:0 negFeatures - fp
      in {tp; fp; fn; tn}

  let tune (models: t list) (pos: playlist) (neg: playlist) (eval: confusion_matrix -> float) : t =
    match models with 
    | [] -> failwith "No models provided"
    | first :: rest -> List.fold ~init:(first, eval @@ test first pos neg) 
                         ~f:(fun (best, score) curr 
                              -> eval @@ test curr pos neg |> fun new_score 
                                 -> if Float.(<) score new_score then (curr, new_score) 
                                 else (best, score)) rest |> fun (best, _) -> best

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