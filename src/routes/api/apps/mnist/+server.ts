import { json, type RequestHandler } from "@sveltejs/kit";

export const POST = (async ({ request, fetch }) => {
  let apiUrl = "https://api.jitx.io/predict";
  let reqestJson = await request.json();

  let array = reqestJson.array;
  let user_id = reqestJson.user_id;

  let req = await fetch(apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ array: array, user_id: user_id }),
  });

  let res = await req.json();
  return json(res);
}) satisfies RequestHandler;

export const GET = (async ({ url, fetch }) => {
  let taskIdFromQuery = url.searchParams.get("task_id");
  let apiUrl = "https://api.jitx.io/predict";
  let req = await fetch(apiUrl + "/" + taskIdFromQuery, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  let res = await req.json();
  return json(res);
}) satisfies RequestHandler;
