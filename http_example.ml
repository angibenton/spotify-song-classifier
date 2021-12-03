open Core
open Lwt
open Cohttp
open Cohttp_lwt_unix
module Np = Np.Numpy

type song = {name: string; id: string; features: Np.Ndarray.t;}

type playlist = {name: string; id: string; features: Np.Ndarray.t;}

(* Returns a promise that will resolve the way f resolves, or with timeout *)
let get_promise_with_timeout ~time ~f =
  Lwt.pick
    [
      (f () >|= fun v -> `Done v)
    ; (Lwt_unix.sleep time >|= fun () -> `Timeout)
    ]
;;

(* json string of an object -> json string of the given field *)
let get_field (field: string) (obj: string): string = 
  obj 
  |> Yojson.Safe.from_string 
  |> Yojson.Safe.Util.member field
  |> Yojson.Safe.to_string 
;;

(* ------- ACCESS TOKEN ------- *)

let token_request_timeout = 20.;; 
let global_token = ref "INITIAL TOKEN - DO NOT USE";;


(* Ask spotify for the server, returns promise of (resp, body) *)
let send_token_request (): (Response.t * Cohttp_lwt.Body.t) t = 
  let uri = Uri.of_string "https://accounts.spotify.com/api/token?grant_type=client_credentials" in
  let headers = Header.init ()
  |> fun h -> Header.add h "Content-Type" "application/x-www-form-urlencoded"
  |> fun h -> Header.add h "Authorization" "Basic ZDgwMGZlMzgwOTAwNGRjZGI1NmViZTkwYjg2ZThlNzQ6ZGM0OTdiMWM2MjAwNDY1Y2JlZmYyNWE5ODhkY2YxYzk="
  in
  Client.call ~headers `POST uri 
;;

(* Handle resolved promise from send_token_request, returns promise of body string *)
let handle_token_response_get_body (resp: Response.t) (body: Cohttp_lwt.Body.t): string t =
  let code = resp |> Response.status |> Code.code_of_status in
  if code <> 200 then 
    Lwt.fail_with @@ "Request for token failed, code: " ^ (Int.to_string code)
  else 
    body |> Cohttp_lwt.Body.to_string 
;;

(* Do the token request with a timeout, handle the response, return a promise of the body string *)
let request_token_return_body () : string t = 
  get_promise_with_timeout ~time:token_request_timeout ~f:send_token_request >>= function
  | `Timeout -> Lwt.fail_with "Timeout expired"
  | `Done (resp, body) -> handle_token_response_get_body resp body 
;;

(* Update the global access token *) 
let update_token () = 
  let body = Lwt_main.run @@ request_token_return_body () in  
  let token = body |> get_field "access_token" |> String.filter ~f:(fun c -> Char.(<>) c '"') in
  global_token := token; 
;;



(* Reasons a (non-token) request could get 401: 


   - expired token
   - ill-formed token (we can make sure this never happen) 
   - bad ID (we have to handle this)
   - more?
   

   Idea: literally check for this exact string?
   {
    "error": {
        "status": 401,
        "message": "The access token expired"
    }
}
*) 


(* ------- REQUESTING SONG DATA ------- *) 

(* TODO: timeouts for song requests? *)

(* TODO: refactor this obvi.  *)

let request_song_features (id: string): string t = 
  let uri = Uri.of_string ("https://api.spotify.com/v1/audio-features/" ^ id) in
  let headers = Header.init ()
  |> fun h -> Header.add h "Content-Type" "application/json"
  |> fun h -> Header.add h "Authorization" ("Bearer " ^ !global_token) 
  |> fun h -> Header.add h "Accept" "application/json"
  in
  Client.call ~headers `GET uri 
  >>= fun (resp, body) ->
  let code = resp |> Response.status |> Code.code_of_status in
  if code = 200 then 
    body |> Cohttp_lwt.Body.to_string
  else 
    Lwt.fail_with @@ "Request for song features failed, code: " ^ (Int.to_string code)
;;

let request_song_metadata (id: string): string t = 
  let uri = Uri.of_string ("https://api.spotify.com/v1/tracks/" ^ id) in
  let headers = Header.init ()
  |> fun h -> Header.add h "Content-Type" "application/json"
  |> fun h -> Header.add h "Authorization" ("Bearer " ^ !global_token) 
  |> fun h -> Header.add h "Accept" "application/json"
  in
  Client.call ~headers `GET uri 
  >>= fun (resp, body) ->
  let code = resp |> Response.status |> Code.code_of_status in
  if code = 200 then 
    body |> Cohttp_lwt.Body.to_string
  else 
    Lwt.fail_with @@ "Request for song metadata failed, code: " ^ (Int.to_string code)
;;

let request_playlist_metadata (id: string): string t = 
  let uri = Uri.of_string ("https://api.spotify.com/v1/playlists/" ^ id ^ "?market=ES") in (* need the market ES? *)
  let headers = Header.init ()
  |> fun h -> Header.add h "Content-Type" "application/json"
  |> fun h -> Header.add h "Authorization" ("Bearer " ^ !global_token) 
  |> fun h -> Header.add h "Accept" "application/json"
  in
  Client.call ~headers `GET uri 
  >>= fun (resp, body) ->
  let code = resp |> Response.status |> Code.code_of_status in
  if code = 200 then 
    body |> Cohttp_lwt.Body.to_string
  else 
    Lwt.fail_with @@ "Request for playlist metadata failed, code: " ^ (Int.to_string code)
;;

let request_song_features_batch (ids_comma_sep: string): string t = 
  let uri = Uri.of_string ("https://api.spotify.com/v1/audio-features/?ids=" ^ ids_comma_sep ^ "?market=ES") in (* need the market ES? *)
  let headers = Header.init ()
  |> fun h -> Header.add h "Content-Type" "application/json"
  |> fun h -> Header.add h "Authorization" ("Bearer " ^ !global_token) 
  |> fun h -> Header.add h "Accept" "application/json"
  in
  Client.call ~headers `GET uri 
  >>= fun (resp, body) ->
  let code = resp |> Response.status |> Code.code_of_status in
  if code = 200 then 
    body |> Cohttp_lwt.Body.to_string
  else 
    Lwt.fail_with @@ "Request for batch of song features failed, code: " ^ (Int.to_string code)
;;

let test_print_stuff_from_a_song (id: string) = 
  let features_body = Lwt_main.run @@ request_song_features id in
  let metadata_body = Lwt_main.run @@ request_song_metadata id in
  let popularity = get_field "popularity" metadata_body in
  let name = get_field "name" metadata_body in
  let danceability = get_field "danceability" features_body in
  print_endline @@ "Name: " ^ name;
  print_endline @@ "Popularity: " ^ popularity;
  print_endline @@ "Danceability: " ^ danceability;
;;


let get_audio_feature (features_yojson: Yojson.Safe.t) (field: string): float = 
  features_yojson
  |> Yojson.Safe.Util.member field
  |> Yojson.Safe.to_string 
  |> Float.of_string 
;;

let feature_names = [
  "danceability";
  "energy";
  "key";
  "loudness";
  "mode";
  "speechiness";
  "acousticness";
  "instrumentalness";
  "liveness";
  "valence";
  "tempo";
  "duration_ms";
  "time_signature";
];;

let features_yojson_to_vector (features_yojson: Yojson.Safe.t): Np.Ndarray.t = 
  List.map feature_names ~f:(get_audio_feature features_yojson) 
  |> Np.Ndarray.of_float_list
  |> Np.reshape ~newshape:[1; (List.length feature_names)]
;;


let song_of_id (id: string): song =
  let features_body = Lwt_main.run @@ request_song_features id in
  let metadata_body = Lwt_main.run @@ request_song_metadata id in
  let name = get_field "name" metadata_body in
  let features = features_body |> Yojson.Safe.from_string |> features_yojson_to_vector  in
  {name; id; features;}
;;

let yojson_extract_list (wrapped_list: Yojson.Safe.t): Yojson.Safe.t list = 
  match wrapped_list with
  | `List ls -> ls |>  List.filter ~f:(fun elem -> match elem with | `Null -> false | _ -> true)
  | _ -> failwith "Yojson object does not have a list at the top level"
;;

let feature_matrix_of_vector_list (vecs: Np.Ndarray.t list): Np.Ndarray.t =  
  let append_vector (cur_matrix: Np.Ndarray.t) (new_vector: Np.Ndarray.t) : Np.Ndarray.t = 
    Np.append ~axis:0 ~arr:cur_matrix () ~values:new_vector
  in
  match vecs with 
  | head :: tail -> 
     List.fold_left tail ~init:head ~f:append_vector 
  | _ -> assert false;
;;

let playlist_of_id (id: string): playlist =
  let playlist_metadata_body = Lwt_main.run @@ request_playlist_metadata id in
  let name = get_field "name" playlist_metadata_body in
  let id_list_comma_sep_string = 
    playlist_metadata_body 
    |> Yojson.Safe.from_string 
    |> Yojson.Safe.Util.member "tracks"
    |> Yojson.Safe.Util.member "items"
    |> yojson_extract_list
    |> List.map ~f:(Yojson.Safe.Util.member "track")
    |> List.map ~f:(Yojson.Safe.Util.member "id")
    |> List.map ~f:Yojson.Safe.to_string
    |> List.map ~f:(String.filter ~f:(fun c -> Char.(<>) c '"'))
    |> String.concat ~sep:"," 
  in
  let batch_song_features_body = Lwt_main.run @@ request_song_features_batch id_list_comma_sep_string in
  print_endline batch_song_features_body;
  print_endline id_list_comma_sep_string; 
  let features = 
    batch_song_features_body 
    |> Yojson.Safe.from_string 
    |> Yojson.Safe.Util.member "audio_features"
    |> yojson_extract_list 
    |> List.map ~f:features_yojson_to_vector
    |> feature_matrix_of_vector_list
  in
  {name; id; features;}
;;

let print_song (s: song): _ = 
  printf "name: %s\n" s.name;
  printf "id: %s\n" s.id;
  printf "features: %s\n" @@ Np.Ndarray.to_string s.features; 
;;

let print_playlist (p: playlist): _ = 
  printf "name: %s\n" p.name;
  printf "id: %s\n" p.id;
  printf "features: %s\n" @@ Np.Ndarray.to_string p.features; 
;;
  

let () =
  print_endline ("Token before update: " ^ !global_token);
  update_token (); 
  print_endline ("Token after update: " ^ !global_token);
  test_print_stuff_from_a_song "5w52BJAqGkV1ewaCVLmjhi";
  test_print_stuff_from_a_song "0ZFBKLOZLIM16RAUb5eomN";
  print_song @@ song_of_id "0ZFBKLOZLIM16RAUb5eomN"; 
  print_playlist @@ playlist_of_id "2vsDLvI5C8p14oQCnpdCkd";
;;