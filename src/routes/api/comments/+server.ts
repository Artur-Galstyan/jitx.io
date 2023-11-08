import { env } from "$env/dynamic/private";
import { prisma } from "$lib/server/prisma.server";
import { COMMENTS_PER_PAGE } from "$lib/utils/constants";
import { json, type RequestHandler } from "@sveltejs/kit";

export const POST = (async ({ request, locals, fetch }) => {
  const session = await locals.auth.validate();
  if (!session) {
    return json({ error: "You must be logged in to comment", status: 401 });
  }

  const user = session.user;
  if (!user) {
    return json({ error: "You must be logged in to comment", status: 401 });
  }

  const requestJson = await request.json();

  const token = requestJson.token;
  if (!token) {
    return json({ error: "Token missing", status: 401 });
  }

  let req = await fetch("https://www.google.com/recaptcha/api/siteverify", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `secret=${env.CAPTCHA_SECRET_KEY}&response=${token}`,
  });

  const captchaResponse = await req.json();
  if (!captchaResponse.success) {
    return json({ error: "Captcha failed", status: 401 });
  }

  const body = requestJson.body;

  if (!body) {
    return json({ error: "Comment must not be empty", status: 400 });
  }

  const postId = requestJson.postId;
  if (!postId) {
    return json({ error: "Post ID must not be empty", status: 400 });
  }

  const newComment = await prisma.comment.create({
    data: {
      body: body,
      postId: postId,
      userId: user.userId as string,
    },
  });

  return json({ comment: newComment });
}) satisfies RequestHandler;

export const DELETE = (async ({ request, locals, fetch }) => {
  const session = await locals.auth.validate();
  if (!session) {
    return json({ error: "You must be logged in to comment", status: 401 });
  }
  const user = session.user;
  if (!user) {
    return json({ error: "You must be logged in to comment", status: 401 });
  }

  const requestJson = await request.json();

  const commentId = requestJson.commentId;

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

  if (comment.userId !== user.userId) {
    return json({
      error: "You do not have permission to delete this comment",
      status: 403,
    });
  }

  await prisma.comment.delete({
    where: {
      id: commentId,
    },
  });

  return json({ deleted: true });
}) satisfies RequestHandler;

export const GET = (async ({ request, locals, fetch, url }) => {
  const searchParams = url.searchParams;

  const postId = searchParams.get("postId");
  if (!postId) {
    return json({ error: "Post ID must not be empty", status: 400 });
  }

  const page = searchParams.get("page");
  if (!page) {
    return json({ error: "Page must not be empty", status: 400 });
  }

  const skip = Math.ceil(COMMENTS_PER_PAGE * (Number(page) - 1));

  const comments = await prisma.comment.findMany({
    where: {
      postId: postId,
    },
    skip: skip,
    take: COMMENTS_PER_PAGE,
    orderBy: {
      createdAt: "desc",
    },
    include: {
      User: {},
      Reactions: {},
    },
  });
  return json({ comments: comments });
}) satisfies RequestHandler;
