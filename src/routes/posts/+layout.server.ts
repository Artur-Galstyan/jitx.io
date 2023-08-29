import { prisma } from "$lib/server/prisma.server";
import type { LayoutServerLoad } from "../$types";

export const load = (async ({ request, locals, url }) => {
    const pathname = url.pathname;
    const slug = pathname.split("/")[2];

    const post = await prisma.post.findMany({
        where: {
            slug: slug
        }
    });

    return { post: post[0] };
}) satisfies LayoutServerLoad;
