// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
    namespace App {
        declare module "*.md";
        // interface Error {}
        // interface Locals {}
        // interface PageData {}
        // interface Platform {}
        interface Session {
            user?: {
                id?: string;
                name?: string | null;
                email?: string | null;
                image?: string | null;
            };
            expires: string; // ISODateString
        }
    }
}

export {};
