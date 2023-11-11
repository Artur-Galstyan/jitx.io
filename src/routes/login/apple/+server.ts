import { dev } from "$app/environment";
import { appleAuth } from "$lib/server/lucia.server";
import type { RequestHandler } from "@sveltejs/kit";

export const GET: RequestHandler = async ({ cookies }) => {
  const [url, state] = await appleAuth.getAuthorizationUrl();
  // store state
  cookies.set("apple_oauth_state", state, {
    httpOnly: true,
    secure: !dev,
    path: "/",
    maxAge: 60 * 60,
  });
  return new Response(null, {
    status: 302,
    headers: {
      Location: url.toString(),
    },
  });
};