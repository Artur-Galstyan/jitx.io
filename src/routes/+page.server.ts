import { prisma } from "$lib/server/prisma.server";
import type { PageServerLoad } from "./$types";

export const load = (async ({ request, locals }) => {
    let posts = await prisma.post.findMany({
        orderBy: {
            createdAt: "desc"
        }
    });

    return { posts: posts };
}) satisfies PageServerLoad;
