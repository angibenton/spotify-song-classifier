open Core
open Lwt
open Cohttp
open Cohttp_lwt_unix
module Np = Np.Numpy

let spotify_api_credentials = "ZDgwMGZlMzgwOTAwNGRjZGI1NmViZTkwYjg2ZThlNzQ6ZGM0OTdiMWM2MjAwNDY1Y2JlZmYyNWE5ODhkY2YxYzk=";;
let spotify_base_uri = "https://api.spotify.com/v1/";;
let token_request_timeout = 20.;; 
let data_request_timeout = 30.;;

type song = {name: string; sid: string; features_vector: Np.Ndarray.t;}

type playlist = {name: string; pid: string; features_matrix: Np.Ndarray.t;}

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
]
;;

module type Spotify = sig 
  (* Generate a (PROMISE OF) new api token to be used for song_of_id and playlist_of_id. Expires in one hour. *)
  val get_new_api_token : _ -> string Lwt.t
  (* Use a song id & access token to query spotify for song data and convert the result to (PROMISE OF) a song object *)
  val song_of_id : string -> string -> song Lwt.t
  (* Use a playlist id & access token to query spotify for playlist data and convert the result to (PROMISE OF) a playlist object *)
  val playlist_of_id : string -> string-> playlist Lwt.t
end


module Spotify : Spotify = struct
  (* --------- HELPERS - HIDDEN ---------- *)

  (* Returns a promise that will resolve the way f resolves, or with timeout *)
  let get_promise_with_timeout ~time ~f =
    Lwt.pick
      [
        (f () >|= fun v -> `Done v)
      ; (Lwt_unix.sleep time >|= fun () -> `Timeout)
      ]

  (* ------- ACCESS TOKEN NETWORK FUNCTIONS ------- *)


  let get_error_msg (body: string): string = 
    body 
    |> Yojson.Safe.from_string 
    |> Yojson.Safe.Util.member "error"
    |> Yojson.Safe.Util.member "message"
    |> Yojson.Safe.to_string
    |> String.filter ~f:(fun c -> Char.(<>) c '"') 


  let send_token_request (): (Response.t * Cohttp_lwt.Body.t) t = 
    let uri = Uri.of_string "https://accounts.spotify.com/api/token?grant_type=client_credentials" in
    let headers = Header.init ()
                  |> fun h -> Header.add h "Content-Type" "application/x-www-form-urlencoded"
                              |> fun h -> Header.add h "Authorization" ("Basic " ^ spotify_api_credentials)
    in
    Client.call ~headers `POST uri 

  let handle_token_response_get_body (resp: Response.t) (body: Cohttp_lwt.Body.t): string t =
    let code = resp |> Response.status |> Code.code_of_status in
    if code <> 200 then 
      body |> Cohttp_lwt.Body.to_string >>= (function body_str -> 
          Lwt.fail_with @@ "Request for token failed: " ^ (Int.to_string code) ^ " " ^ (get_error_msg body_str))
    else 
      body |> Cohttp_lwt.Body.to_string 

  let request_token_return_body () : string t = 
    get_promise_with_timeout ~time:token_request_timeout ~f:send_token_request >>= function
    | `Timeout -> Lwt.fail_with "Timeout expired"
    | `Done (resp, body) -> handle_token_response_get_body resp body 



  (* ------- SONG DATA NETWORK FUNCTIONS ------- *) 

  let general_api_request (uri_suffix: string) (description: string) (api_token: string): string t =
    let do_request (): (Response.t * Cohttp_lwt.Body.t) t = 
      let uri = Uri.of_string @@ spotify_base_uri ^ uri_suffix in
      let headers = 
        Header.init ()
        |> fun h -> Header.add h "Content-Type" "application/json"
        |> fun h -> Header.add h "Authorization" ("Bearer " ^ api_token) 
        |> fun h -> Header.add h "Accept" "application/json"
      in
      Client.call ~headers `GET uri
    in
    let handle_response = fun resp body -> 
      let code = resp |> Response.status |> Code.code_of_status in
      if code <> 200 then
        Cohttp_lwt.Body.to_string body >>= (fun body_str -> 
          Lwt.fail_with @@ description ^ " failed: " ^ (Int.to_string code) ^ " " ^ (get_error_msg body_str)
        )
      else 
        Cohttp_lwt.Body.to_string body                                                                
    in
    get_promise_with_timeout ~f:do_request ~time:data_request_timeout >>= (function 
      | `Timeout -> Lwt.fail_with @@ description ^ " failed: timeout" 
      | `Done (resp, body) -> handle_response resp body 
    )


  let request_song_metadata (id: string) (api_token: string): string t = 
    let uri_suffix = "tracks/" ^ id in
    let description = "request metadata for song (id = " ^ id ^ ")" in
    general_api_request uri_suffix description api_token

  let request_song_features (id: string) (api_token: string): string t = 
    let uri_suffix = "audio-features/" ^ id in
    let description = "request features for song (id = " ^ id ^ ")" in
    general_api_request uri_suffix description api_token 


  let request_playlist_metadata (id: string) (api_token: string): string t = 
    let uri_suffix = "playlists/" ^ id ^ "?market=ES" in
    let description = "request metadata for playlist (id = " ^ id ^ ")" in
    general_api_request uri_suffix description api_token


  let request_song_features_batch (ids_comma_sep: string) (api_token: string): string t = 
    let uri_suffix = "audio-features/?ids=" ^ ids_comma_sep ^ "?market=ES" in
    let description = "request batch of song ids" in
    general_api_request uri_suffix description api_token


  (* -------- Yojson and (Yojson --> Numpy) manipulation -------- *)

  let get_field_remove_quotes (field: string) (obj: string): string = 
    obj 
    |> Yojson.Safe.from_string 
    |> Yojson.Safe.Util.member field
    |> Yojson.Safe.to_string 
    |> String.filter ~f:(fun c -> Char.(<>) c '"') 

  let get_audio_feature (features_yojson: Yojson.Safe.t) (field: string): float = 
    features_yojson
    |> Yojson.Safe.Util.member field
    |> Yojson.Safe.to_string 
    |> Float.of_string 

  let yojson_extract_list (wrapped_list: Yojson.Safe.t): Yojson.Safe.t list = 
    match wrapped_list with
    | `List ls -> ls |>  List.filter ~f:(fun elem -> match elem with | `Null -> false | _ -> true)
    | _ -> failwith "Yojson object does not have a list at the top level"

  let matrix_of_vector_list (vecs: Np.Ndarray.t list): Np.Ndarray.t =  
    let append_vector (cur_matrix: Np.Ndarray.t) (new_vector: Np.Ndarray.t) : Np.Ndarray.t = 
      Np.append ~axis:0 ~arr:cur_matrix () ~values:new_vector
    in
    match vecs with 
    | head :: tail -> 
      List.fold_left tail ~init:head ~f:append_vector 
    | _ -> assert false

  let song_features_yojson_to_vector (features_yojson: Yojson.Safe.t): Np.Ndarray.t = 
    List.map feature_names ~f:(get_audio_feature features_yojson) 
    |> Np.Ndarray.of_float_list
    |> Np.reshape ~newshape:[1; (List.length feature_names)]

  let playlist_features_yojson_to_matrix (features_yojson: Yojson.Safe.t): Np.Ndarray.t = 
    features_yojson
    |> Yojson.Safe.Util.member "audio_features"
    |> yojson_extract_list 
    |> List.map ~f:song_features_yojson_to_vector
    |> matrix_of_vector_list

  let ids_from_playlist_body (playlist_metadata_body: string): string =
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

    
  (* --------- EXPOSED API --------- *)

  let get_new_api_token _ : string Lwt.t = 
    let%lwt body = request_token_return_body () in  
    let token = get_field_remove_quotes "access_token" body in
    Lwt.return @@ token 

  let playlist_of_id (pid: string) (api_token: string): playlist Lwt.t =
    let%lwt playlist_body = request_playlist_metadata pid api_token in
    let name = get_field_remove_quotes "name" playlist_body in
    let ids = ids_from_playlist_body playlist_body in
    let%lwt batch_body = request_song_features_batch ids api_token in
    let features_matrix = batch_body |> Yojson.Safe.from_string |> playlist_features_yojson_to_matrix in
    Lwt.return {name; pid; features_matrix;}

  let song_of_id (sid: string) (api_token: string): song Lwt.t =
    let%lwt features_body = request_song_features sid api_token in
    let%lwt metadata_body = request_song_metadata sid api_token in
    let name = get_field_remove_quotes "name" metadata_body in
    let features_vector = features_body |> Yojson.Safe.from_string |> song_features_yojson_to_vector in
    Lwt.return {name; sid; features_vector;}
end