import { prisma } from "$lib/server/prisma.server";
import { redirect, type Actions } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";

export const load = (async ({ request, locals }) => {
    const session = await locals.getSession();
    if (!session) {
        return redirect(303, "/whatwereyouthinking");
    }

    if (!session.user) {
        return redirect(303, "/whatwereyouthinking");
    }

    const posts = await prisma.post.findMany();

    return { posts: posts };
}) satisfies PageServerLoad;

export const actions = {
    create: async ({ request, locals }) => {
        const session = await locals.getSession();
        if (!session) {
            return redirect(303, "/whatwereyouthinking");
        }
        if (!session.user) {
            return redirect(303, "/whatwereyouthinking");
        }

        const formData = await request.formData();

        try {
            let tags =
                formData.get("tags") === ""
                    ? []
                    : JSON.parse(formData.get("tags") as string);
            const newPost = await prisma.post.create({
                data: {
                    title: formData.get("title") as string,
                    slug: formData.get("slug") as string,
                    tags: tags
                }
            });

            return { newPost: newPost };
        } catch (error) {
            return { error: error };
        }
    },
    delete: async ({ request, locals }) => {
        const session = await locals.getSession();
        if (!session) {
            return redirect(303, "/whatwereyouthinking");
        }
        if (!session.user) {
            return redirect(303, "/whatwereyouthinking");
        }

        const formData = await request.formData();

        try {
            await prisma.post.delete({
                where: {
                    id: parseInt(formData.get("id") as string)
                }
            });

            return { success: true };
        } catch (error) {
            return { error: error };
        }
    }
} satisfies Actions;
