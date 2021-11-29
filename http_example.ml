open Core
open Lwt
open Cohttp
open Cohttp_lwt_unix

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
  else if code = 401 then
    Lwt.fail_with @@ "TODO HANDLE 401 features"
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
  else if code = 401 then
    Lwt.fail_with @@ "TODO HANDLE 401 metadata"
  else 
    Lwt.fail_with @@ "Request for song metadata failed, code: " ^ (Int.to_string code)
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


let () =
  print_endline ("Token before update: " ^ !global_token);
  update_token (); 
  print_endline ("Token after update: " ^ !global_token);
  test_print_stuff_from_a_song "5w52BJAqGkV1ewaCVLmjhi";
  test_print_stuff_from_a_song "0ZFBKLOZLIM16RAUb5eomN";
;;