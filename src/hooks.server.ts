import type { Handle } from "@sveltejs/kit";
import { auth } from "$lib/server/lucia.server";

export const handle: Handle = async ({ event, resolve }) => {
  // we can pass `event` because we used the SvelteKit middleware
  event.locals.auth = auth.handleRequest(event);
  return resolve(event);
};
