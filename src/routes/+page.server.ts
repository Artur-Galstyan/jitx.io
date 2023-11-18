import { prisma } from "$lib/server/prisma.server";
import { POSTS_PER_PAGE } from "$lib/utils/constants";
import type { PageServerLoad } from "./$types";

export const load = (async ({ request, locals, params, url }) => {
  const take = url.searchParams.get("take") || POSTS_PER_PAGE.toString();
  const skip = url.searchParams.get("skip") || "0";
  const search = url.searchParams.get("search") || "";

  let posts = null;
  if (search !== "") {
    posts = await prisma.post.findMany({
      where: {
        NOT: {
          status: "HIDDEN",
        },
        OR: [{ title: { search: search.split(" ").join(" | ") } }],
      },
      orderBy: {
        updatedAt: "desc",
      },
      take: parseInt(take),
      skip: parseInt(skip),
    });
  } else {
    posts = await prisma.post.findMany({
      where: {
        NOT: {
          status: "HIDDEN",
        },
      },
      orderBy: {
        updatedAt: "desc",
      },
      take: parseInt(take),
      skip: parseInt(skip),
    });
  }

  let projects = await prisma.project.findMany({
    orderBy: {
      createdAt: "desc",
    },
  });

  return {
    posts: posts,
    projects: projects,
    totalPosts: await prisma.post.count({}),
  };
}) satisfies PageServerLoad;
