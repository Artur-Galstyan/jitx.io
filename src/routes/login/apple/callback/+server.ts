import { appleAuth, auth } from "$lib/server/lucia.server";
import { OAuthRequestError } from "@lucia-auth/oauth";
import type { RequestHandler } from "@sveltejs/kit";

export const POST = (async ({ request, url, locals, cookies }) => {
  console.log("Apple callback", url.searchParams);
  console.log("Apple request text", await request.text());
  console.log("Apple request body", await request.json());
  const code = url.searchParams.get("code");

  // validate state
  if (!code) {
    console.log("Error in apple callback, no code");
    return new Response(null, {
      status: 400,
    });
  }
  try {
    const { getExistingUser, appleUser, createUser } =
      await appleAuth.validateCallback(code);

    const getUser = async () => {
      const userJSON = url.searchParams.get("user");
      let email = null;
      if (userJSON) {
        const user = JSON.parse(userJSON);
        email = user.email;
      }
      const existingUser = await getExistingUser();
      if (existingUser) return existingUser;
      return await createUser({
        // @ts-ignore
        attributes: {
          username: appleUser.sub,
          email: appleUser.email ?? email,
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
