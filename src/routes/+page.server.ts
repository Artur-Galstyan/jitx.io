import { prisma } from "$lib/server/prisma.server";
import type { PageServerLoad } from "./$types";

export const load = (async ({ request, locals }) => {
    let posts = await prisma.post.findMany({
        orderBy: {
            createdAt: "desc"
        }
    });

    let projects = await prisma.project.findMany({
        orderBy: {
            createdAt: "desc"
        }
    });

    return { posts: posts, projects: projects };
}) satisfies PageServerLoad;
