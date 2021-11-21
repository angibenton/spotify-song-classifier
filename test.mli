open Torch

type song = {name: string; id: string; features: Tensor.t;}

type playlist = {name: string; id: string; features: Tensor.t;}
  
module type Model = sig 
  (* the model *)
  type t 

  (* save a model into a file with a give filename *)
  val save : t -> string -> unit
  (* open up a model file with a given filename, parse a model object from it *)
  val load : string -> t

  (* train a binary classifier on two playlists represented as tensors *)
  val train : playlist -> playlist -> t
  (* classify a song represented by a vector into one of the two playlists true = first, false = second *)
  val classify : t -> song -> playlist 

end 

module type SpotifyAPI = sig 

  (* use an id to query spotify for song metadata and convert the result to a song object *)
  val song_of_id : string -> song 
  (* use an id to query spotify for playlist metadata and convert the result to a playlist object *)
  val playlist_of_id : string -> playlist 
  
end