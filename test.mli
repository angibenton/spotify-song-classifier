open Torch
module Np = Np.Numpy

type song = {name: string; id: string; features: Np.Ndarray.t;}

type playlist = {name: string; id: string; features: Np.Ndarray.t;}

type confusion_matrix = {tp: int; fp: int; tn: int; fn: int}

type svm = {hyperplane: Np.Ndarray.t; class1: string; class2: string}

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
end 

module type Classification = sig
  type classifier
  (* classify a song represented by a vector into one of the two playlists *)
  val classify : classifier -> song -> playlist
  (* return the confusion matrix from testing the model on a tensor of labeled songs *)
  val test : classifier -> Np.Ndarray.t -> confusion_matrix
  val accuracy : confusion_matrix -> float
  val f1_score : confusion_matrix -> float
end

module type SpotifyAPI = sig 

  (* use an id to query spotify for song metadata and convert the result to a song object *)
  val song_of_id : string -> song 
  (* use an id to query spotify for playlist metadata and convert the result to a playlist object *)
  val playlist_of_id : string -> playlist 
  
end