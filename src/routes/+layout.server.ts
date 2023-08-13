import type { LayoutServerLoad } from "./";

export const load = (async ({ locals }) => {
    return {
        session: await locals.getSession()
    };
}) satisfies LayoutServerLoad;
