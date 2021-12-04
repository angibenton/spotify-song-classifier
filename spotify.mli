module Np = Np.Numpy

type song = {name: string; id: string; features: Np.Ndarray.t;}

type playlist = {name: string; id: string; features: Np.Ndarray.t;}

module type Spotify = sig 
  (* Generate a (PROMISE OF) new api token to be used for song_of_id and playlist_of_id. Expires in one hour. *)
  val get_new_api_token : _ -> string Lwt.t
  (* Use a song id & access token to query spotify for song data and convert the result to (PROMISE OF) a song object *)
  val song_of_id : string -> string -> song Lwt.t
  (* Use a playlist id & access token to query spotify for playlist data and convert the result to (PROMISE OF) a playlist object *)
  val playlist_of_id : string -> string-> playlist Lwt.t
end