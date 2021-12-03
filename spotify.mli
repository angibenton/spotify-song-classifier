module type SpotifyAPI = sig 

  (* use an id to query spotify for song metadata and convert the result to a song object *)
  val song_of_id : string -> song 
  (* use an id to query spotify for playlist metadata and convert the result to a playlist object *)
  val playlist_of_id : string -> playlist 
  
end