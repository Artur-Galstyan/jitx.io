import { prisma } from "$lib/server/prisma.server";
import { ReactionType } from "@prisma/client";
import { json, type RequestHandler } from "@sveltejs/kit";

export const POST = (async ({ request, locals, params }) => {
    const session = await locals.getSession();
    if (!session) {
        return json({ error: "You must be logged in to comment", status: 401 });
    }

    const user = session.user;
    if (!user) {
        return json({ error: "You must be logged in to comment", status: 401 });
    }

    const commentId = params.id;

    if (!commentId) {
        return json({ error: "Comment ID must not be empty", status: 400 });
    }

    const comment = await prisma.comment.findUnique({
        where: {
            id: commentId
        }
    });

    if (!comment) {
        return json({ error: "Comment not found", status: 404 });
    }
    let reactionTypeFromJson = (await request.json()).reactionType;

    if (reactionTypeFromJson === "up") {
        reactionTypeFromJson = ReactionType.LIKE;
    } else if (reactionTypeFromJson === "down") {
        reactionTypeFromJson = ReactionType.DISLIKE;
    } else if (reactionTypeFromJson === "heart") {
        reactionTypeFromJson = ReactionType.HEART;
    } else if (reactionTypeFromJson === "clap") {
        reactionTypeFromJson = ReactionType.CLAP;
    } else if (reactionTypeFromJson === "fire") {
        reactionTypeFromJson = ReactionType.FIRE;
    } else if (reactionTypeFromJson === "sad") {
        reactionTypeFromJson = ReactionType.SAD;
    } else if (reactionTypeFromJson === "party") {
        reactionTypeFromJson = ReactionType.PARTY;
    }
    // Check if user has already upvoted this comment
    const existingReaction = await prisma.reaction.findFirst({
        where: {
            commentId: commentId,
            userId: user.id,
            type: reactionTypeFromJson
        }
    });

    if (existingReaction) {
        // User has already upvoted this comment, so remove the upvote
        await prisma.reaction.delete({
            where: {
                id: existingReaction.id
            }
        });

        return json({ status: 200 });
    }

    // Create a new upvote
    await prisma.reaction.create({
        data: {
            commentId: commentId,
            userId: user.id,
            type: reactionTypeFromJson
        }
    });

    return json({ status: 200 });
}) satisfies RequestHandler;
