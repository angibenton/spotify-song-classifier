module Np = Np.Numpy
open Core
open Spotify 

(* TESTER FOR THE SPOTIFY MODULE *)

(* TODO actual tests *)

let print_song (s: Spotify.song): _ = 
  printf "name: %s\n" s.name;
  printf "id: %s\n" s.sid;
  printf "features: %s\n" @@ Np.Ndarray.to_string s.features_vector; 
;;

let print_playlist (p: Spotify.playlist): _ = 
  printf "name: %s\n" p.name;
  printf "id: %s\n" p.pid;
  printf "features: %s\n" @@ Np.Ndarray.to_string p.features_matrix; 
;;


let () =
  let p = 
    let%lwt token = Spotify_api.get_new_api_token () in
    let%lwt song1 = Spotify_api.song_of_id "5e9TFTbltYBg2xThimr0rU" token in 
    let%lwt song2 = Spotify_api.song_of_id "1HzDhHApjdjXPLHF6GGYhu" token in 
    let%lwt song3 = Spotify_api.song_of_id "4U45aEWtQhrm8A5mxPaFZ7" token in 
    let%lwt playlist = Spotify_api.playlist_of_id "01keRwFGHF7Rw1wnPwbyB1" token in 
    print_endline ("Token: " ^ token);
    print_playlist playlist; 
    print_song song1; 
    print_song song2; 
    print_song song3; 
    Lwt.return ()
  in
  Lwt_main.run p;
;;