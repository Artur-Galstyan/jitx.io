import { prisma } from "$lib/server/prisma.server";
import { ReactionType } from "@prisma/client";
import { json, type RequestHandler } from "@sveltejs/kit";

export const POST = (async ({ request, locals, params }) => {
  const session = await locals.auth.validate();
  if (!session) {
    return json({
      error: "You must be logged in to make a reaction!",
      status: 401,
    });
  }

  const user = session.user;
  if (!user) {
    return json({
      error: "You must be logged in to make a reaction!",
      status: 401,
    });
  }

  const commentId = params.id;

  if (!commentId) {
    return json({ error: "Comment ID must not be empty", status: 400 });
  }

  const comment = await prisma.comment.findUnique({
    where: {
      id: commentId,
    },
  });

  if (!comment) {
    return json({ error: "Comment not found", status: 404 });
  }
  let reactionTypeFromJson = (await request.json()).reactionType;

  if (reactionTypeFromJson === "LIKE") {
    reactionTypeFromJson = ReactionType.LIKE;
  } else if (reactionTypeFromJson === "DISLIKE") {
    reactionTypeFromJson = ReactionType.DISLIKE;
  } else if (reactionTypeFromJson === "HEART") {
    reactionTypeFromJson = ReactionType.HEART;
  } else if (reactionTypeFromJson === "CLAP") {
    reactionTypeFromJson = ReactionType.CLAP;
  } else if (reactionTypeFromJson === "FIRE") {
    reactionTypeFromJson = ReactionType.FIRE;
  } else if (reactionTypeFromJson === "SAD") {
    reactionTypeFromJson = ReactionType.SAD;
  } else if (reactionTypeFromJson === "PARTY") {
    reactionTypeFromJson = ReactionType.PARTY;
  }
  // Check if user has already upvoted this comment
  const existingReaction = await prisma.reaction.findFirst({
    where: {
      commentId: commentId,
      userId: user.userId,
      type: reactionTypeFromJson,
    },
  });

  if (existingReaction) {
    // User has already upvoted this comment, so remove the upvote
    await prisma.reaction.delete({
      where: {
        id: existingReaction.id,
      },
    });

    return json({ status: 200 });
  }

  // Create a new upvote
  await prisma.reaction.create({
    data: {
      commentId: commentId,
      userId: user.userId,
      type: reactionTypeFromJson,
    },
  });

  return json({ status: 200 });
}) satisfies RequestHandler;
