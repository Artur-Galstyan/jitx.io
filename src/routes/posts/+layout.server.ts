import { prisma } from "$lib/server/prisma.server";
import { COMMENTS_PER_PAGE } from "$lib/utils/constants";
import type { LayoutServerLoad } from "../$types";

export const load = (async ({ request, locals, url }) => {
    const pathname = url.pathname;
    const slug = pathname.split("/")[2];

    const post = await prisma.post.findMany({
        where: {
            slug: slug
        }
    });

    if (!post) {
        return {
            status: 404,
            error: new Error(`Post with slug "${slug}" not found`)
        };
    }

    const comments = await prisma.comment.findMany({
        where: {
            postId: post[0].id
        },
        take: COMMENTS_PER_PAGE,
        orderBy: {
            createdAt: "desc"
        },
        include: {
            User: {},
            Reactions: {}
        }
    });

    const allComments = await prisma.comment.findMany({});

    const totalPages = allComments.length / COMMENTS_PER_PAGE;

    return { post: post[0], comments: comments, totalPages: totalPages };
}) satisfies LayoutServerLoad;
