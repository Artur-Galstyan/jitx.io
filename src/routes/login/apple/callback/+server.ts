import { appleAuth, auth } from "$lib/server/lucia.server";
import { OAuthRequestError } from "@lucia-auth/oauth";
import type { RequestHandler } from "@sveltejs/kit";

export const GET = (async ({ url, locals, cookies }) => {
  const storedState = cookies.get("apple_oauth_state");

  const state = url.searchParams.get("state");
  const code = url.searchParams.get("code");
  // validate state
  if (!storedState || !state || storedState !== state || !code) {
    console.log("Error in apple callback");
    return new Response(null, {
      status: 400,
    });
  }
  try {
    const { getExistingUser, appleUser, createUser } =
      await appleAuth.validateCallback(code);

    const getUser = async () => {
      const existingUser = await getExistingUser();
      if (existingUser) return existingUser;
      return await createUser({
        // @ts-ignore
        attributes: {
          username: appleUser.sub,
          email: appleUser.email ?? null,
        },
      });
    };

    const user = await getUser();
    const session = await auth.createSession({
      userId: user.userId,
      attributes: {},
    });
    locals.auth.setSession(session);
    return new Response(null, {
      status: 302,
      headers: {
        Location: "/",
      },
    });
  } catch (e) {
    console.log("Error in apple callback ", e);
    if (e instanceof OAuthRequestError) {
      // invalid code
      return new Response(null, {
        status: 400,
      });
    }
    return new Response(null, {
      status: 500,
    });
  }
}) satisfies RequestHandler;
