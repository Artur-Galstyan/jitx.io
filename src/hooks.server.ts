import { env } from "$env/dynamic/private";
import { prisma } from "$lib/server/prisma.server";
import type { Provider } from "@auth/core/providers";
import Discord from "@auth/core/providers/discord";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { SvelteKitAuth } from "@auth/sveltekit";
import { type Handle, redirect } from "@sveltejs/kit";
import { sequence } from "@sveltejs/kit/hooks";

async function authorization({ event, resolve }) {
    // Protect any routes under /authenticated
    if (event.url.pathname.startsWith("/dashboard")) {
        const session = await event.locals.getSession();

        if (!session) {
            throw redirect(303, "/whatwereyouthinking");
        }

        if (!session.user) {
            throw redirect(303, "/whatwereyouthinking");
        }

        if (session.user.email !== env.DISCORD_ADMIN_EMAIL) {
            throw redirect(303, "/whatwereyouthinking");
        }
    }

    // If the request is still here, just proceed as normally
    return resolve(event);
}

export const handle = sequence(
    SvelteKitAuth({
        providers: [
            Discord({
                clientId: env.DISCORD_CLIENT_ID,
                clientSecret: env.DISCORD_CLIENT_SECRET
            })
        ] as Provider[],
        adapter: PrismaAdapter(prisma),
        callbacks: {
            session: async ({ session, user }) => {
                return {
                    ...session,
                    user: user
                };
            }
        }
    }),
    authorization
) satisfies Handle;
